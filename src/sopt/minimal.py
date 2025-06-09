import os
import torch
import tqdm
import threading
import queue
import random
from collections import defaultdict
import numpy as np
import tarfile
import time

from torchao.optim import _AdamW

from impl import (
  get_model, tkn, MODEL_SIZE, bytewise_detokenize, bytewise_tokenize,
   GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES,  CHECKPOINT_EVERY,  HOMEDIR, BATCH_SIZES)
from src.sopt.impl import PRINT_STATS_EVERY
from util import report_cuda_size, timeit

CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

def report_model_size_with_tokens(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    params_M = params / 1_000_000
    target_tokens = int(params_M * 4000)

    min_tokens = 16_000
    max_tokens = 4_000_000

    target_tokens = max(min_tokens, min(max_tokens, target_tokens))

    print(f"Model: {params // 1_000_000}M parameters ({params:,} total)")
    print(f"Target tokens per update: {target_tokens:,} (4K per 1M params)")

    return params, target_tokens

def token_based_training_step(model, data_iter, optimizer, target_tokens_per_update,
                            bucket_usage_stats, bucket_training_counts, batch_size_stats):

    model.train()
    optimizer.zero_grad()

    accumulated_tokens = 0
    accumulated_loss = 0
    num_micro_batches = 0
    total_actual_tokens = 0

    while accumulated_tokens < target_tokens_per_update:
        src, src_mask, tgt, bucket_len, current_batch_size, actual_tokens = next(data_iter)

        bucket_usage_stats[bucket_len] += 1
        bucket_training_counts[bucket_len] += 1
        batch_size_stats[current_batch_size] += 1

        src = src.to('cuda', non_blocking=True)
        src_mask = src_mask.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)

        loss = model(src, tgt, mask=src_mask)

        token_weight = actual_tokens / target_tokens_per_update
        weighted_loss = loss * token_weight

        weighted_loss.backward()

        accumulated_tokens += actual_tokens
        accumulated_loss += loss.item() * token_weight
        total_actual_tokens += actual_tokens
        num_micro_batches += 1

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return accumulated_loss, accumulated_tokens, num_micro_batches, total_actual_tokens

def save_checkpoint(model, optim):
    print("saving", CHECKPOINT)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }, CHECKPOINT)

def load_checkpoint(model, optim):
    if os.path.exists(CHECKPOINT):
        print(f"loading {CHECKPOINT}")
        checkpoint = torch.load(CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optim
    return model, optim


class FixedBucketRandomCollector:
    def __init__(self, batch_sizes_dict, max_enc_len=ENC_SEQ_LEN, max_dec_len=DEC_SEQ_LEN):
        self.batch_sizes_dict = batch_sizes_dict
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        self.bucket_lengths = sorted(batch_sizes_dict.keys())

        self.buckets = {}
        self.bucket_batch_sizes = {}

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
        for bucket_len in self.bucket_lengths:
            if actual_length <= bucket_len:
                return bucket_len
        return None

    def add_sample(self, src_tokens, src_mask, tgt_tokens):
        pad_token = tkn('PAD')  # 256
        src_actual_len = len([t for t in src_tokens if t != pad_token])
        tgt_actual_len = len([t for t in tgt_tokens if t != pad_token])

        max_actual_len = max(src_actual_len, tgt_actual_len)
        bucket_len = self.get_bucket_for_length(max_actual_len)

        if bucket_len is None:
            return None

        sample = {
            'src': src_tokens,
            'src_mask': src_mask,
            'tgt': tgt_tokens,
            'src_actual_len': src_actual_len,
            'tgt_actual_len': tgt_actual_len
        }

        self.buckets[bucket_len].append(sample)
        return None

    def get_next_batch_random(self):
        ready_buckets = []
        weights = []

        for bucket_len in self.bucket_lengths:
            target_batch_size = self.bucket_batch_sizes[bucket_len]
            available_samples = len(self.buckets[bucket_len])

            if available_samples >= target_batch_size:
                ready_buckets.append(bucket_len)
                num_full_batches = available_samples // target_batch_size
                weights.append(num_full_batches)

        if not ready_buckets:
            return None

        selected_bucket = random.choices(ready_buckets, weights=weights)[0]
        return self.create_batch_from_bucket(selected_bucket)

    def create_batch_from_bucket(self, bucket_len):
        target_batch_size = self.bucket_batch_sizes[bucket_len]
        samples = self.buckets[bucket_len][:target_batch_size]
        self.buckets[bucket_len] = self.buckets[bucket_len][target_batch_size:]

        batch_src, batch_src_mask, batch_tgt = [], [], []
        total_actual_tokens = 0

        for sample in samples:
            actual_src_len = len(sample['src'])
            actual_tgt_len = len(sample['tgt'])
            total_actual_tokens += actual_src_len + actual_tgt_len

            src = sample['src']
            tgt = sample['tgt']

            if len(src) > bucket_len or len(tgt) > bucket_len:
                print(f"WARNING: Sample exceeds bucket size - src:{len(src)}, tgt:{len(tgt)}, bucket:{bucket_len}")
                continue

            pad_token = tkn('PAD')  # 256
            if len(src) < bucket_len:
                src_mask = [True] * len(src) + [False] * (bucket_len - len(src))
                src.extend([pad_token] * (bucket_len - len(src)))
            else:
                src_mask = [True] * bucket_len

            if len(tgt) < bucket_len:
                tgt.extend([pad_token] * (bucket_len - len(tgt)))

            batch_src.append(src)
            batch_src_mask.append(src_mask)
            batch_tgt.append(tgt)

        return {
            'src': torch.tensor(batch_src).long().pin_memory(),
            'src_mask': torch.tensor(batch_src_mask).bool().pin_memory(),
            'tgt': torch.tensor(batch_tgt).long().pin_memory(),
            'bucket_len': bucket_len,
            'batch_size': len(batch_src),
            'actual_tokens': total_actual_tokens
        }

    def get_pending_batches(self):
        batches = []
        for bucket_len in self.bucket_lengths:
            if len(self.buckets[bucket_len]) > 0:
                samples = self.buckets[bucket_len]
                self.buckets[bucket_len] = []

                if samples:
                    original_batch_size = self.bucket_batch_sizes[bucket_len]
                    self.bucket_batch_sizes[bucket_len] = len(samples)
                    batch = self.create_batch_from_bucket(bucket_len)
                    self.bucket_batch_sizes[bucket_len] = original_batch_size

                    if batch:
                        batches.append(batch)

        return batches

    def get_bucket_stats(self):
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
        print("\n=== Bucket Status ===")
        for bucket_len in self.bucket_lengths:
            count = len(self.buckets[bucket_len])
            target = self.bucket_batch_sizes[bucket_len]
            fill_pct = (count / target) * 100 if target > 0 else 0
            ready = "READY" if count >= target else "waiting"
            print(f"  {bucket_len:4d} tokens: {count:3d}/{target:2d} samples ({fill_pct:5.1f}% full) [{ready}]")


class FullDatasetLoader:
    """Async loader that processes binary .o files from single corpus tar.gz archives."""

    def __init__(self, data_dir, batch_collector, prefetch_buffer=1000, num_workers=2):
        self.data_dir = data_dir
        self.batch_collector = batch_collector
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = num_workers

        self.batch_queue = queue.Queue(maxsize=prefetch_buffer)

        self.stop_event = threading.Event()
        self.epoch_complete = threading.Event()

        # Get all corpus archives
        self.corpus_archives = self._get_corpus_archives()
        print(f"Found {len(self.corpus_archives)} corpus archives")

        self.loader_thread = threading.Thread(target=self._loader_worker)
        self.loader_thread.daemon = True
        self.loader_thread.start()

    def _get_corpus_archives(self):
        all_files = os.listdir(self.data_dir)
        corpus_files = [f for f in all_files if f.startswith('corpus_') and f.endswith('.tar.gz')]

        corpus_paths = []
        for corpus_file in corpus_files:
            corpus_path = os.path.join(self.data_dir, corpus_file)
            corpus_paths.append(corpus_path)

        corpus_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        return corpus_paths

    def _process_sample(self, enc_bytes, dec_bytes):
        #print(len(enc_bytes), len(dec_bytes))
        unopt_tokens = bytewise_tokenize(enc_bytes)
        opt_tokens = bytewise_tokenize(dec_bytes)

        # Skip if sequences would be too long
        # Source: no special tokens added
        # Target: DECSTART + EOS tokens will be added (2 tokens)
        if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) + 2 >= DEC_SEQ_LEN:
            return None

        opt_tokens_with_special = [tkn('DECSTART')] + opt_tokens + [tkn('EOS')]

        return {
            'src': unopt_tokens,  # No special tokens
            'src_mask': None,     # Will be created during batching
            'tgt': opt_tokens_with_special  # Has DECSTART + original + EOS
        }

    def _extract_and_pair_files(self, corpus_archive):
        encoder_files = {}
        decoder_files = {}

        with tarfile.open(corpus_archive, 'r:gz') as tar:
            for member in tar.getmembers():
                if not member.name.endswith('.o'):
                    raise
                file_obj = tar.extractfile(member)
                if not file_obj:
                    raise
                file_data = file_obj.read()

                if member.name.startswith('encoder_'):
                        index = int(member.name.split('_')[1].split('.')[0])
                        encoder_files[index] = file_data
                elif member.name.startswith('decoder_'):
                        index = int(member.name.split('_')[1].split('.')[0])
                        decoder_files[index] = file_data

        pairs = []
        common_indices = set(encoder_files.keys()) & set(decoder_files.keys())
        for index in sorted(common_indices):
            pairs.append((encoder_files[index], decoder_files[index]))

        return pairs


    def _loader_worker(self):
        epoch = 0
        while not self.stop_event.is_set():
            print(f"Starting epoch {epoch + 1}, processing {len(self.corpus_archives)} corpus archives...")

            archives = self.corpus_archives.copy()
            random.shuffle(archives)

            samples_loaded = 0
            samples_skipped = 0
            batches_created = 0

            for file_idx, corpus_archive in enumerate(archives):
                if self.stop_event.is_set():
                    break

                print(f"Loading corpus archive {file_idx + 1}/{len(archives)}: {os.path.basename(corpus_archive)}")

                try:
                    file_pairs = self._extract_and_pair_files(corpus_archive)

                    for enc_bytes, dec_bytes in file_pairs:
                        if self.stop_event.is_set():
                            break

                        sample = self._process_sample(enc_bytes, dec_bytes)

                        if sample is None:
                            samples_skipped += 1
                            continue

                        self.batch_collector.add_sample(
                            sample['src'], sample['src_mask'], sample['tgt']
                        )

                        samples_loaded += 1

                        batch = self.batch_collector.get_next_batch_random()
                        if batch is not None:
                            try:
                                self.batch_queue.put(batch, timeout=1)
                                batches_created += 1
                            except queue.Full:
                                if not self.stop_event.is_set():
                                    self.batch_queue.put(batch)
                                    batches_created += 1

                except Exception as e:
                    print(f"Error loading {corpus_archive}: {e}")
                    continue

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

            self.epoch_complete.set()
            self.epoch_complete.clear()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = self.batch_queue.get(timeout=30)
            return (batch['src'], batch['src_mask'], batch['tgt'],
                   batch['bucket_len'], batch['batch_size'], batch['actual_tokens'])
        except queue.Empty:
            if self.stop_event.is_set():
                raise StopIteration
            else:
                print("Warning: Batch queue empty but loader should be running")
                raise StopIteration

    def stop(self):
        print("Stopping data loader...")
        self.stop_event.set()
        self.loader_thread.join(timeout=5)

    def get_queue_size(self):
        return self.batch_queue.qsize()


def format_token_rates(total_tokens, elapsed_time):
    if elapsed_time == 0:
        return {
            'per_sec': '0',
            'per_min': '0',
            'per_hour': '0',
            'per_day': '0'
        }

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
    data_dir = f"{HOMEDIR}/superopt_data/"

    batch_collector = FixedBucketRandomCollector(
        batch_sizes_dict=BATCH_SIZES,
        max_enc_len=ENC_SEQ_LEN,
        max_dec_len=DEC_SEQ_LEN
    )

    # Calculate prefetch buffer based on batch sizes
    base_prefetch_buffer = 4
    max_batch_size = max(BATCH_SIZES.values())
    prefetch_buffer = base_prefetch_buffer * max_batch_size

    print(f"Prefetch buffer calculation:")
    print(f"  Base prefetch buffer: {base_prefetch_buffer}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Calculated prefetch buffer: {prefetch_buffer}")

    print(f"  Per-bucket effective capacity:")
    for bucket_len, batch_size in sorted(BATCH_SIZES.items()):
        effective_samples = prefetch_buffer * batch_size
        print(f"    {bucket_len:4d} tokens: {prefetch_buffer} batches × {batch_size} samples = {effective_samples:,} samples")

    data_loader = FullDatasetLoader(
        data_dir,
        batch_collector,
        prefetch_buffer=prefetch_buffer,
        num_workers=16
    )

    data_iter = iter(data_loader)

    model = get_model(tkn('PAD'))

    params, target_tokens_per_update = report_model_size_with_tokens(model)

    optim = _AdamW(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)

    model, optim = load_checkpoint(model, optim)

    print(f"Starting training with token-based gradient accumulation")
    print(f"  Buckets: {batch_collector.bucket_lengths}")
    print(f"  Target tokens per update: {target_tokens_per_update:,}")
    print(f"  Learning rate: {LEARNING_RATE}")

    start_time = time.time()
    iteration_count = 0
    total_tokens_processed = 0
    losses = []
    bucket_usage_stats = defaultdict(int)
    bucket_training_counts = defaultdict(int)
    batch_size_stats = defaultdict(int)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='\ntraining'):

        loss, tokens_in_step, micro_batches, actual_tokens = token_based_training_step(
            model, data_iter, optim, target_tokens_per_update,
            bucket_usage_stats, bucket_training_counts, batch_size_stats
        )
        losses.append(loss)

        total_tokens_processed += actual_tokens
        iteration_count += micro_batches

        elapsed_time = time.time() - start_time
        iterations_per_sec = iteration_count / elapsed_time if elapsed_time > 0 else 0
        token_rates = format_token_rates(total_tokens_processed, elapsed_time)

        print(f'\nTokens: {total_tokens_processed:,} total | {token_rates["per_sec"]} tok/s | {token_rates["per_min"]} tok/min | {token_rates["per_hour"]} tok/hr | {token_rates["per_day"]} tok/day')
        print(f'{i}: loss={loss:.4f} | {iterations_per_sec:.2f} iter/s | micro_batches={micro_batches} | step_tokens={tokens_in_step:,}')
        print(f"avg loss {sum(losses) / (i+1):.4f}")

        if i % PRINT_STATS_EVERY == 0 and i > 0:
            print(f"  Queue size: {data_loader.get_queue_size()}")
            print("  Bucket training counts:")
            total_trained = sum(bucket_training_counts.values())
            if total_trained > 0:
                for bucket_len in sorted(batch_collector.bucket_lengths):
                    count = bucket_training_counts.get(bucket_len, 0)
                    percentage = (count / total_trained) * 100
                    target_batch_size = batch_collector.bucket_batch_sizes[bucket_len]
                    total_samples_trained = count * target_batch_size
                    print(f"    {bucket_len:4d} tokens: {count:4d} batches ({percentage:5.1f}%) = {total_samples_trained:,} samples")

        report_cuda_size()

        if i % CHECKPOINT_EVERY == 0 and i > 0:
            save_checkpoint(model, optim)

        if i % GENERATE_EVERY == 0 and i > 0:
            model.eval()

            src, src_mask, tgt, _, _, _ = next(data_iter)
            src = src[:1].to('cuda', non_blocking=True)
            src_mask = src_mask[:1].to('cuda', non_blocking=True)
            tgt = tgt[:1].to('cuda', non_blocking=True)

            start_tokens = torch.tensor([tkn('DECSTART')]).to('cuda')
            sample = model.generate(src, start_tokens, DEC_SEQ_LEN, eos_token=tkn('EOS'), mask=src_mask)

            def filter_byte_tokens(tokens):
                return [t for t in tokens if 0 <= t < 256]

            print_stmt = ""
            print_stmt += f"\ninput (first 100 bytes):  {bytewise_detokenize(filter_byte_tokens(src.tolist()[0][:100]))}\n"
            print_stmt += f"predicted (first 100 bytes):  {bytewise_detokenize(filter_byte_tokens(sample.tolist()[:100]))}\n"
            print_stmt += f"actual (first 100 bytes):     {bytewise_detokenize(filter_byte_tokens(tgt.tolist()[0][:100]))}\n"
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
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    main()