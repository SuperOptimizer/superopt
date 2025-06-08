import os
import gzip
import torch
import tqdm
import sentencepiece as spm
from torch.optim import AdamW
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import random
from collections import defaultdict
import math
import numpy as np

from torchao.optim import _AdamW, AdamW4bit
from torchao.optim import CPUOffloadOptimizer

from impl import (
  get_model, tokenize_bytes, detokenize_bytes, tokenize_hexstr, detokenize_hexstr, tkn, MODEL_SIZE,
   GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES, TMP, CHECKPOINT_EVERY, HOMEDIR, BATCH_SIZES)
from src.sopt.impl import PRINT_STATS_EVERY
from util import report_cuda_size, timeit, chunkify
from codegen import gen_yarpgen

CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

def calculate_target_tokens_per_update(num_parameters):
    """
    Simple scaling: 4K tokens per 1M parameters

    Args:
        num_parameters: Total number of trainable parameters

    Returns:
        target_tokens_per_update: Tokens to accumulate before optimizer step
    """
    params_M = num_parameters / 1_000_000
    target_tokens = int(params_M * 4000)

    # Apply reasonable bounds
    min_tokens = 16_000    # Minimum for stability
    max_tokens = 4_000_000 # Maximum for memory

    return max(min_tokens, min(max_tokens, target_tokens))

def report_model_size_with_tokens(model):
    """Enhanced version that calculates target token accumulation"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    target_tokens = calculate_target_tokens_per_update(params)

    print(f"Model: {params // 1_000_000}M parameters ({params:,} total)")
    print(f"Target tokens per update: {target_tokens:,} (4K per 1M params)")

    return params, target_tokens

def token_based_training_step(model, data_iter, optimizer, target_tokens_per_update,
                            bucket_usage_stats, bucket_training_counts, batch_size_stats):
    """
    Single training step using token-based gradient accumulation
    """
    model.train()
    optimizer.zero_grad()

    accumulated_tokens = 0
    accumulated_loss = 0
    num_micro_batches = 0
    total_actual_tokens = 0

    while accumulated_tokens < target_tokens_per_update:
        try:
            src, src_mask, tgt, bucket_len, current_batch_size, actual_tokens = next(data_iter)

            # Track statistics
            bucket_usage_stats[bucket_len] += 1
            bucket_training_counts[bucket_len] += 1
            batch_size_stats[current_batch_size] += 1

            # Move to GPU
            src = src.to('cuda', non_blocking=True)
            src_mask = src_mask.to('cuda', non_blocking=True)
            tgt = tgt.to('cuda', non_blocking=True)

            # Forward pass
            loss = model(src, tgt, mask=src_mask)

            # Weight by token contribution - this replaces reference_batch_size scaling
            token_weight = actual_tokens / target_tokens_per_update
            weighted_loss = loss * token_weight

            # Backward pass
            weighted_loss.backward()

            # Track metrics
            accumulated_tokens += actual_tokens
            accumulated_loss += loss.item() * token_weight
            total_actual_tokens += actual_tokens
            num_micro_batches += 1

        except StopIteration:
            # End of data - step with what we have
            break

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return accumulated_loss, accumulated_tokens, num_micro_batches, total_actual_tokens

def save_checkpoint(model, optim, loss):
    print("saving", CHECKPOINT)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss.item(),
    }, CHECKPOINT)

def load_checkpoint(model, optim, loss=0):
    if os.path.exists(CHECKPOINT):
        print(f"loading {CHECKPOINT}")
        checkpoint = torch.load(CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    return model, optim, loss


class FixedBucketRandomCollector:
    """Collects samples into fixed buckets (1k, 2k, 4k, 8k) with predefined batch sizes and weighted random selection."""

    def __init__(self, batch_sizes_dict, max_enc_len=ENC_SEQ_LEN, max_dec_len=DEC_SEQ_LEN):
        self.batch_sizes_dict = batch_sizes_dict
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        # Get bucket lengths from the batch sizes dictionary
        self.bucket_lengths = sorted(batch_sizes_dict.keys())

        # Storage for samples
        self.buckets = {}
        self.bucket_batch_sizes = {}

        # Set up buckets and batch sizes
        for bucket_len in self.bucket_lengths:
            self.bucket_batch_sizes[bucket_len] = batch_sizes_dict[bucket_len]
            self.buckets[bucket_len] = []

        print(f"Fixed bucket weighted random collector initialized:")
        print(f"  Selection: Weighted by number of available batches per bucket")
        print(f"  Bucket batch sizes:")
        for bucket_len in self.bucket_lengths:
            batch_size = self.bucket_batch_sizes[bucket_len]
            print(f"    {bucket_len:4d} tokens: batch_size={batch_size:3d}")

    def get_bucket_for_length(self, actual_length):
        """Get the appropriate bucket for a given sequence length, or None if too large."""
        for bucket_len in self.bucket_lengths:
            if actual_length <= bucket_len:
                return bucket_len
        # Return None if sequence is too large for any bucket
        return None

    def add_sample(self, src_tokens, src_mask, tgt_tokens):
        """Add a sample to the appropriate bucket, or skip if too large."""
        # Get actual lengths
        src_actual_len = len([t for t in src_tokens if t != tkn('PAD')])
        tgt_actual_len = len([t for t in tgt_tokens if t != tkn('PAD')])

        # Find bucket based on max length needed
        max_actual_len = max(src_actual_len, tgt_actual_len)
        bucket_len = self.get_bucket_for_length(max_actual_len)

        # Skip sample if it doesn't fit in any bucket
        if bucket_len is None:
            # Optionally log skipped samples (but not too frequently to avoid spam)
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            self._skip_count += 1

            if self._skip_count <= 10 or self._skip_count % 1000 == 0:
                print(f"Skipping sample {self._skip_count}: src_len={src_actual_len}, tgt_len={tgt_actual_len}, max_len={max_actual_len} (exceeds largest bucket {max(self.bucket_lengths)})")

            return None

        # Debug: Print first few samples to verify bucketing logic
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0

        if self._debug_count < 5:  # Show first 5 samples
            print(f"Sample {self._debug_count + 1}: src_len={src_actual_len}, tgt_len={tgt_actual_len}, max_len={max_actual_len} → bucket={bucket_len}")
            self._debug_count += 1

        sample = {
            'src': src_tokens,
            'src_mask': src_mask,
            'tgt': tgt_tokens,
            'src_actual_len': src_actual_len,
            'tgt_actual_len': tgt_actual_len
        }

        self.buckets[bucket_len].append(sample)

        # Don't return batches immediately - let weighted random selection decide
        return None

    def get_next_batch_random(self):
        """Get the next batch using weighted random selection from ready buckets."""
        # Find all buckets that have enough samples and calculate weights
        ready_buckets = []
        weights = []

        for bucket_len in self.bucket_lengths:
            target_batch_size = self.bucket_batch_sizes[bucket_len]
            available_samples = len(self.buckets[bucket_len])

            if available_samples >= target_batch_size:
                ready_buckets.append(bucket_len)
                # Weight by how many full batches can be made from this bucket
                num_full_batches = available_samples // target_batch_size
                weights.append(num_full_batches)

        # If no buckets are ready, return None
        if not ready_buckets:
            return None

        # Weighted random selection (buckets with more samples get higher probability)
        selected_bucket = random.choices(ready_buckets, weights=weights)[0]
        return self.create_batch_from_bucket(selected_bucket)

    def create_batch_from_bucket(self, bucket_len):
        """Create a batch from the specified bucket."""
        target_batch_size = self.bucket_batch_sizes[bucket_len]
        samples = self.buckets[bucket_len][:target_batch_size]
        self.buckets[bucket_len] = self.buckets[bucket_len][target_batch_size:]

        batch_src, batch_src_mask, batch_tgt = [], [], []
        total_actual_tokens = 0  # Count actual tokens before padding

        for sample in samples:
            # Count actual tokens before any padding
            actual_src_len = len(sample['src'])
            actual_tgt_len = len(sample['tgt'])
            total_actual_tokens += actual_src_len + actual_tgt_len

            src = sample['src']
            tgt = sample['tgt']

            # Samples should fit in bucket (skipping logic handles oversized samples)
            # But add a safety check just in case
            if len(src) > bucket_len or len(tgt) > bucket_len:
                print(f"WARNING: Sample exceeds bucket size - src:{len(src)}, tgt:{len(tgt)}, bucket:{bucket_len}")
                continue

            # Pad to bucket length (no trimming needed if bucketing is correct)
            if len(src) < bucket_len:
                src_mask = [True] * len(src) + [False] * (bucket_len - len(src))
                src.extend([tkn('PAD')] * (bucket_len - len(src)))
            else:
                src_mask = [True] * bucket_len

            if len(tgt) < bucket_len:
                tgt.extend([tkn('PAD')] * (bucket_len - len(tgt)))

            batch_src.append(src)
            batch_src_mask.append(src_mask)
            batch_tgt.append(tgt)

        return {
            'src': torch.tensor(batch_src).long().pin_memory(),
            'src_mask': torch.tensor(batch_src_mask).bool().pin_memory(),
            'tgt': torch.tensor(batch_tgt).long().pin_memory(),
            'bucket_len': bucket_len,
            'batch_size': len(batch_src),
            'actual_tokens': total_actual_tokens  # Actual tokens before padding
        }

    def get_pending_batches(self):
        """Get any remaining partial batches at end of epoch."""
        batches = []
        for bucket_len in self.bucket_lengths:
            if len(self.buckets[bucket_len]) > 0:
                # Create batch with whatever samples remain
                samples = self.buckets[bucket_len]
                self.buckets[bucket_len] = []

                if samples:
                    # Temporarily override batch size
                    original_batch_size = self.bucket_batch_sizes[bucket_len]
                    self.bucket_batch_sizes[bucket_len] = len(samples)
                    batch = self.create_batch_from_bucket(bucket_len)
                    self.bucket_batch_sizes[bucket_len] = original_batch_size  # Restore

                    if batch:
                        batches.append(batch)

        return batches

    def get_bucket_stats(self):
        """Get statistics about current bucket usage."""
        stats = {}
        total_samples = 0
        for bucket_len in self.bucket_lengths:
            count = len(self.buckets[bucket_len])
            target_size = self.bucket_batch_sizes[bucket_len]
            total_samples += count
            if count > 0:
                stats[bucket_len] = {
                    'samples': count,
                    'target_batch_size': target_size,
                    'fill_percentage': (count / target_size) * 100
                }
        return stats, total_samples

    def print_bucket_status(self):
        """Print current status of all buckets."""
        print("\n=== Bucket Status ===")
        for bucket_len in self.bucket_lengths:
            count = len(self.buckets[bucket_len])
            target = self.bucket_batch_sizes[bucket_len]
            fill_pct = (count / target) * 100 if target > 0 else 0
            ready = "READY" if count >= target else "waiting"
            print(f"  {bucket_len:4d} tokens: {count:3d}/{target:2d} samples ({fill_pct:5.1f}% full) [{ready}]")


class FullDatasetLoader:
    """Async loader that goes through the entire dataset before repeating."""

    def __init__(self, data_dir, sp_model_path, batch_collector, prefetch_buffer=1000, num_workers=2):
        self.data_dir = data_dir
        self.sp_model_path = sp_model_path
        self.batch_collector = batch_collector
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = num_workers

        # Queue for processed batches
        self.batch_queue = queue.Queue(maxsize=prefetch_buffer)

        # Control events
        self.stop_event = threading.Event()
        self.epoch_complete = threading.Event()

        # Get all file pairs
        self.file_pairs = self._get_file_pairs()
        print(f"Found {len(self.file_pairs)} file pairs")

        # Start background loading
        self.loader_thread = threading.Thread(target=self._loader_worker)
        self.loader_thread.daemon = True
        self.loader_thread.start()

    def _get_file_pairs(self):
        """Get all encoder/decoder file pairs."""
        num_gzips = len(os.listdir(self.data_dir))
        assert num_gzips % 2 == 0

        file_pairs = []
        for i in range(num_gzips // 2):
            encoder_gzip = f"{self.data_dir}/encoder_corpus_{i}.txt.gzip"
            decoder_gzip = f"{self.data_dir}/decoder_corpus_{i}.txt.gzip"
            file_pairs.append((encoder_gzip, decoder_gzip))

        return file_pairs

    def _process_sample(self, sp, enc_line, dec_line):
        """Process a single sample and return tokens WITHOUT padding to max length."""
        unopt_tokens = tokenize_hexstr(sp, enc_line)
        opt_tokens = tokenize_hexstr(sp, dec_line)

        # Skip if sequences would be too long
        # Source: no special tokens added
        # Target: DECSTART + EOS tokens will be added (2 tokens)
        if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) + 2 >= DEC_SEQ_LEN:
            return None

        # Prepare the tokens - add special tokens to target only
        opt_tokens_with_special = [tkn('DECSTART')] + opt_tokens + [tkn('EOS')]

        return {
            'src': unopt_tokens,  # No special tokens
            'src_mask': None,     # Will be created during batching
            'tgt': opt_tokens_with_special  # Has DECSTART + original + EOS
        }

    def _loader_worker(self):
        """Background worker that loads data from all files."""
        sp = spm.SentencePieceProcessor(model_file=self.sp_model_path)

        epoch = 0
        while not self.stop_event.is_set():
            print(f"Starting epoch {epoch + 1}, processing {len(self.file_pairs)} file pairs...")

            # Shuffle file order each epoch for better randomization
            file_pairs = self.file_pairs.copy()
            random.shuffle(file_pairs)

            samples_loaded = 0
            samples_skipped = 0
            batches_created = 0

            for file_idx, (enc_file, dec_file) in enumerate(file_pairs):
                if self.stop_event.is_set():
                    break

                print(f"Loading file pair {file_idx + 1}/{len(file_pairs)}: {os.path.basename(enc_file)}")

                try:
                    with gzip.open(enc_file, 'rt') as f, gzip.open(dec_file, 'rt') as g:
                        for line_num, (enc_line, dec_line) in enumerate(zip(f, g)):
                            if self.stop_event.is_set():
                                break

                            # Process the sample
                            sample = self._process_sample(sp, enc_line.strip(), dec_line.strip())

                            if sample is None:
                                samples_skipped += 1
                                continue

                            # Add to batch collector (doesn't return batch immediately)
                            self.batch_collector.add_sample(
                                sample['src'], sample['src_mask'], sample['tgt']
                            )

                            samples_loaded += 1

                            # Check for ready batches using weighted random selection
                            batch = self.batch_collector.get_next_batch_random()
                            if batch is not None:
                                try:
                                    self.batch_queue.put(batch, timeout=1)
                                    batches_created += 1
                                except queue.Full:
                                    if not self.stop_event.is_set():
                                        self.batch_queue.put(batch)  # Block until space available
                                        batches_created += 1

                            # Print progress occasionally
                            if samples_loaded % 10000 == 0:
                                bucket_stats, pending_samples = self.batch_collector.get_bucket_stats()
                                print(f"  Loaded {samples_loaded} samples, {batches_created} batches created, {pending_samples} pending")
                                if samples_loaded % 50000 == 0:
                                    self.batch_collector.print_bucket_status()

                except Exception as e:
                    print(f"Error loading {enc_file}: {e}")
                    continue

            # Handle any remaining partial batches at end of epoch
            remaining_batches = self.batch_collector.get_pending_batches()
            for batch in remaining_batches:
                try:
                    self.batch_queue.put(batch, timeout=1)
                    batches_created += 1
                except queue.Full:
                    if not self.stop_event.is_set():
                        self.batch_queue.put(batch)
                        batches_created += 1

            print(f"Epoch {epoch + 1} complete: {samples_loaded} samples loaded, {samples_skipped} skipped, {batches_created} batches created")
            epoch += 1

            # Signal that we've completed a full pass through the dataset
            self.epoch_complete.set()
            self.epoch_complete.clear()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next batch from the queue."""
        try:
            batch = self.batch_queue.get(timeout=30)  # 30 second timeout
            return (batch['src'], batch['src_mask'], batch['tgt'],
                   batch['bucket_len'], batch['batch_size'], batch['actual_tokens'])
        except queue.Empty:
            if self.stop_event.is_set():
                raise StopIteration
            else:
                print("Warning: Batch queue empty but loader should be running")
                raise StopIteration

    def stop(self):
        """Stop the background loader."""
        print("Stopping data loader...")
        self.stop_event.set()
        self.loader_thread.join(timeout=5)

    def get_queue_size(self):
        """Get current queue size for monitoring."""
        return self.batch_queue.qsize()


def count_tokens_in_batch(src, tgt, pad_token_id):
    """Count non-padding tokens in source and target tensors."""
    src_tokens = (src != pad_token_id).sum().item()
    tgt_tokens = (tgt != pad_token_id).sum().item()
    return src_tokens + tgt_tokens

def format_token_rates(total_tokens, elapsed_time):
    """Calculate and format token processing rates."""
    if elapsed_time == 0:
        return "0 tokens/sec"

    tokens_per_sec = total_tokens / elapsed_time
    tokens_per_min = tokens_per_sec * 60
    tokens_per_hour = tokens_per_sec * 3600
    tokens_per_day = tokens_per_sec * 86400

    return {
        'per_sec': f"{tokens_per_sec:.0f}",
        'per_min': f"{tokens_per_min:.0f}",
        'per_hour': f"{tokens_per_hour:.0f}",
        'per_day': f"{tokens_per_day:.0f}"
    }

@timeit
def train():
    sp_model_path = f'{ROOTDIR}/misc/superopt.model'
    data_dir = f"{HOMEDIR}/superopt_data/"

    # Create fixed bucket weighted random collector
    batch_collector = FixedBucketRandomCollector(
        batch_sizes_dict=BATCH_SIZES,
        max_enc_len=ENC_SEQ_LEN,
        max_dec_len=DEC_SEQ_LEN
    )

    # Calculate prefetch buffer based on batch sizes
    #todo: make this better. this will collect base_prefetch_buffer * the max
    # for any, so if we batch 80 512ctx calls, then we'll store 80 8k ones, which is fine for
    # now i guess but it could be better
    base_prefetch_buffer = 1  # Base buffer size for theoretical batch size of 1
    max_batch_size = max(BATCH_SIZES.values())
    prefetch_buffer = base_prefetch_buffer * max_batch_size

    print(f"Prefetch buffer calculation:")
    print(f"  Base prefetch buffer: {base_prefetch_buffer}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Calculated prefetch buffer: {prefetch_buffer}")

    # Show prefetch buffer for each bucket type
    print(f"  Per-bucket effective capacity:")
    for bucket_len, batch_size in sorted(BATCH_SIZES.items()):
        effective_samples = prefetch_buffer * batch_size
        print(f"    {bucket_len:4d} tokens: {prefetch_buffer} batches × {batch_size} samples = {effective_samples:,} samples")

    data_loader = FullDatasetLoader(
        data_dir,
        sp_model_path,
        batch_collector,
        prefetch_buffer=prefetch_buffer,
        num_workers=16
    )

    data_iter = iter(data_loader)

    model = get_model(tkn('PAD'))

    # Replace report_model_size(model) with token-aware version
    params, target_tokens_per_update = report_model_size_with_tokens(model)

    optim = _AdamW(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=100)
    model, optim, loss = load_checkpoint(model, optim)

    print(f"Starting training with token-based gradient accumulation")
    print(f"  Buckets: {batch_collector.bucket_lengths}")
    print(f"  Target tokens per update: {target_tokens_per_update:,}")

    # Start timing and iteration counting
    import time
    start_time = time.time()
    iteration_count = 0
    total_tokens_processed = 0
    losses = []
    bucket_usage_stats = defaultdict(int)
    bucket_training_counts = defaultdict(int)  # Track how many batches trained per bucket
    batch_size_stats = defaultdict(int)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='\ntraining'):

        # NEW: Token-based training step
        loss, tokens_in_step, micro_batches, actual_tokens = token_based_training_step(
            model, data_iter, optim, target_tokens_per_update,
            bucket_usage_stats, bucket_training_counts, batch_size_stats
        )

        # Update tracking
        total_tokens_processed += actual_tokens
        iteration_count += micro_batches

        # Calculate and report rates
        elapsed_time = time.time() - start_time
        iterations_per_sec = iteration_count / elapsed_time
        token_rates = format_token_rates(total_tokens_processed, elapsed_time)

        print(f'\nTokens: {total_tokens_processed:,} total | {token_rates["per_sec"]} tok/s | {token_rates["per_min"]} tok/min | {token_rates["per_hour"]} tok/hr | {token_rates["per_day"]} tok/day')
        print(f'{i}: loss={loss:.4f} | {iterations_per_sec:.2f} iter/s | micro_batches={micro_batches} | step_tokens={tokens_in_step:,}')
        losses.append(loss)
        print(f"avg loss {sum(losses) / (i+1):.4f}")

        if i % PRINT_STATS_EVERY == 0 and i > 0:
            print(f"  Queue size: {data_loader.get_queue_size()}")

            print("  Bucket usage distribution:")
            total_bucket_uses = sum(bucket_usage_stats.values())
            if total_bucket_uses > 0:
                for bucket_len in sorted(bucket_usage_stats.keys()):
                    count = bucket_usage_stats[bucket_len]
                    percentage = (count / total_bucket_uses) * 100
                    print(f"    {bucket_len:4d} tokens: {count:4d} batches ({percentage:5.1f}%)")

            print("  Bucket training counts (total batches trained with token-based accumulation):")
            total_trained = sum(bucket_training_counts.values())
            if total_trained > 0:
                for bucket_len in sorted(batch_collector.bucket_lengths):
                    count = bucket_training_counts.get(bucket_len, 0)
                    percentage = (count / total_trained) * 100 if total_trained > 0 else 0
                    target_batch_size = batch_collector.bucket_batch_sizes[bucket_len]
                    total_samples_trained = count * target_batch_size
                    print(f"    {bucket_len:4d} tokens: {count:4d} batches ({percentage:5.1f}%) = {total_samples_trained:,} samples")

            print("  Batch size distribution:")
            total_batches = sum(batch_size_stats.values())
            if total_batches > 0:
                for batch_size in sorted(batch_size_stats.keys()):
                    count = batch_size_stats[batch_size]
                    percentage = (count / total_batches) * 100
                    print(f"    batch_size={batch_size:2d}: {count:4d} batches ({percentage:5.1f}%)")

        scheduler.step(i/NUM_BATCHES)
        report_cuda_size()

        if i % CHECKPOINT_EVERY == 0 and i > 0:
            save_checkpoint(model, optim, loss)

        if i % GENERATE_EVERY == 0 and i > 0:
            model.eval()
            # Get a sample for generation
            src, src_mask, tgt, _, _, _ = next(data_iter)
            src = src[:1].to('cuda', non_blocking=True)  # Take first sample
            src_mask = src_mask[:1].to('cuda', non_blocking=True)
            tgt = tgt[:1].to('cuda', non_blocking=True)

            start_tokens = torch.tensor([tkn('DECSTART')]).to('cuda')
            sample = model.generate(src, start_tokens, DEC_SEQ_LEN, eos_token=tkn('EOS'), mask=src_mask)

            sp = spm.SentencePieceProcessor(model_file=sp_model_path)
            print_stmt = ""
            print_stmt += f"\ninput tokenized:  \n{detokenize_bytes(sp, src.tolist()[0])} \n"
            print_stmt += f"\npredicted detokenized:  \n{detokenize_bytes(sp, sample.tolist())}\n"
            print_stmt += f"\nactual detokenized:     \n{detokenize_bytes(sp, tgt.tolist()[0])}\n"
            print(print_stmt)

    # Clean up
    data_loader.stop()


def main():
    train()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_fp8 = True
    torch.backends.cuda.enable_flash_sdp(True)  # Ensure flash attention
    torch.backends.cuda.enable_math_sdp(False)   # Disable slower attention
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory-efficient but slower attention
    main()