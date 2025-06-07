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

from torchao.optim import _AdamW, AdamW4bit
from torchao.optim import CPUOffloadOptimizer

from impl import (
  get_model, tokenize_bytes, detokenize_bytes, tokenize_hexstr, detokenize_hexstr, tkn, MODEL_SIZE, BATCH_SIZE,
   GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES, TMP, CHECKPOINT_EVERY, GRADIENT_ACCUMULATE_EVERY, HOMEDIR)
from util import report_cuda_size, timeit, report_model_size, chunkify
from codegen import gen_yarpgen

CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

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


class FullDatasetLoader:
    """Async loader that goes through the entire dataset before repeating."""

    def __init__(self, data_dir, sp_model_path, prefetch_buffer=1000, num_workers=2):
        self.data_dir = data_dir
        self.sp_model_path = sp_model_path
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = num_workers

        # Queue for processed samples
        self.sample_queue = queue.Queue(maxsize=prefetch_buffer)

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
        """Process a single sample and return tensors or None if invalid."""
        unopt_tokens = tokenize_hexstr(sp, enc_line)
        opt_tokens = tokenize_hexstr(sp, dec_line)

        # Skip if sequences are too long
        if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
            return None

        # Prepare the tokens
        opt_tokens.insert(0, tkn('DECSTART'))
        opt_tokens.append(tkn('EOS'))

        mask = [True] * len(unopt_tokens)
        mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
        unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
        opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))

        return {
            'src': torch.tensor([unopt_tokens]).long().pin_memory(),
            'src_mask': torch.tensor([mask]).bool().pin_memory(),
            'tgt': torch.tensor([opt_tokens]).long().pin_memory()
        }

    def _loader_worker(self):
        """Background worker that loads data from all files."""
        sp = spm.SentencePieceProcessor(model_file=self.sp_model_path)

        epoch = 0
        while not self.stop_event.is_set():
            print(f"Starting epoch {epoch + 1}, processing {len(self.file_pairs)} file pairs...")

            # Optional: shuffle file order each epoch for better randomization
            file_pairs = self.file_pairs.copy()
            random.shuffle(file_pairs)  # Uncomment if you want random file order

            samples_loaded = 0
            samples_skipped = 0

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

                            # Put sample in queue (this will block if queue is full)
                            try:
                                self.sample_queue.put(sample, timeout=1)
                                samples_loaded += 1

                                # Print progress occasionally
                                if samples_loaded % 10000 == 0:
                                    print(f"  Loaded {samples_loaded} samples from file {file_idx + 1}")

                            except queue.Full:
                                # Training is slower than loading, which is good
                                if not self.stop_event.is_set():
                                    self.sample_queue.put(sample)  # Block until space available
                                    samples_loaded += 1

                except Exception as e:
                    print(f"Error loading {enc_file}: {e}")
                    continue

            print(f"Epoch {epoch + 1} complete: {samples_loaded} samples loaded, {samples_skipped} skipped")
            epoch += 1

            # Signal that we've completed a full pass through the dataset
            self.epoch_complete.set()
            self.epoch_complete.clear()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next sample from the queue."""
        try:
            sample = self.sample_queue.get(timeout=30)  # 30 second timeout
            return sample['src'], sample['src_mask'], sample['tgt']
        except queue.Empty:
            if self.stop_event.is_set():
                raise StopIteration
            else:
                # This shouldn't happen unless there's an issue with the loader
                print("Warning: Queue empty but loader should be running")
                raise StopIteration

    def stop(self):
        """Stop the background loader."""
        print("Stopping data loader...")
        self.stop_event.set()
        self.loader_thread.join(timeout=5)

    def get_queue_size(self):
        """Get current queue size for monitoring."""
        return self.sample_queue.qsize()



def collect_batch(data_iter, batch_size):
    """Collect multiple samples from data iterator and concatenate into a batch."""
    batch_src, batch_mask, batch_tgt = [], [], []

    for _ in range(batch_size):
        src, src_mask, tgt = next(data_iter)
        batch_src.append(src)
        batch_mask.append(src_mask)
        batch_tgt.append(tgt)

    return (torch.cat(batch_src, dim=0),
            torch.cat(batch_mask, dim=0),
            torch.cat(batch_tgt, dim=0))

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
def train():  # Add configurable batch size parameter
    sp_model_path = f'{ROOTDIR}/misc/superopt.model'
    data_dir = f"{HOMEDIR}/superopt_data/"

    # Choose loading strategy
    use_full_async = True  # True for queue-based, False for buffered

    if ENC_SEQ_LEN == 8192:
        prefetch_buffer = 256
    elif ENC_SEQ_LEN == 4096:
        prefetch_buffer = 1024
    elif ENC_SEQ_LEN == 2048:
        prefetch_buffer = 8192
    elif ENC_SEQ_LEN == 1024:
        prefetch_buffer = 32768

    data_loader = FullDatasetLoader(
        data_dir,
        sp_model_path,
        prefetch_buffer=prefetch_buffer,  # How many samples to keep in queue
        num_workers=16
    )

    data_iter = iter(data_loader)

    model = get_model(tkn('PAD'))
    report_model_size(model)
    optim = _AdamW(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=100)
    model, optim, loss = load_checkpoint(model, optim)

    # Prime the iterator
    next(data_iter)

    print(f"Starting training with batch size {BATCH_SIZE} and full dataset async loading...")

    # Start timing and iteration counting
    import time
    start_time = time.time()
    iteration_count = 0
    total_tokens_processed = 0
    losses = []

    # Helper function to safely collect batch with iterator restart handling
    def safe_collect_batch(data_iter, BATCH_SIZE):
        batch_src, batch_mask, batch_tgt = [], [], []

        for _ in range(BATCH_SIZE):
            try:
                src, src_mask, tgt = next(data_iter)
                batch_src.append(src)
                batch_mask.append(src_mask)
                batch_tgt.append(tgt)
            except StopIteration:
                # Handle iterator exhaustion - restart and continue collecting
                print("Data iterator exhausted during batch collection, restarting...")
                data_iter = iter(data_loader)
                src, src_mask, tgt = next(data_iter)
                batch_src.append(src)
                batch_mask.append(src_mask)
                batch_tgt.append(tgt)

        return (torch.cat(batch_src, dim=0),
                torch.cat(batch_mask, dim=0),
                torch.cat(batch_tgt, dim=0)), data_iter

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='\ntraining'):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            # Collect a batch of samples
            (src, src_mask, tgt), data_iter = safe_collect_batch(data_iter, BATCH_SIZE)

            # Count tokens in this batch (before moving to GPU)
            batch_tokens = count_tokens_in_batch(src, tgt, tkn('PAD'))
            total_tokens_processed += batch_tokens

            # Move to GPU
            src = src.to('cuda', non_blocking=True)
            src_mask = src_mask.to('cuda', non_blocking=True)
            tgt = tgt.to('cuda', non_blocking=True)

            loss = model(src, tgt, mask=src_mask)
            (loss / (GRADIENT_ACCUMULATE_EVERY * BATCH_SIZE)).backward()

            iteration_count += 1

        # Calculate and report iterations per second
        elapsed_time = time.time() - start_time
        iterations_per_sec = iteration_count / elapsed_time
        token_rates = format_token_rates(total_tokens_processed, elapsed_time)
        print(f'\nTokens: {total_tokens_processed:,} total | {token_rates["per_sec"]} tok/s | {token_rates["per_min"]} tok/min | {token_rates["per_hour"]} tok/hr | {token_rates["per_day"]} tok/day')
        print(f'\n{i}: {loss.item():.4f} | {iterations_per_sec:.2f} iter/s | batch_size: {BATCH_SIZE}')
        losses.append(loss.item())
        print(f"avg loss {sum(losses) / (i+1)}")

        # Optional: print queue size for monitoring
        if use_full_async and i % 100 == 0:
            print(f"  Queue size: {data_loader.get_queue_size()}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        scheduler.step(i/NUM_BATCHES)
        optim.zero_grad()

        if i % CHECKPOINT_EVERY == 0:
            report_cuda_size()
            if i > 0:
                save_checkpoint(model, optim, loss)

        if i % GENERATE_EVERY == 0:
            model.eval()
            # Get a sample for generation (just use first sample from a batch)
            (src, src_mask, tgt), data_iter = safe_collect_batch(data_iter, 1)
            src = src.to('cuda', non_blocking=True)
            src_mask = src_mask.to('cuda', non_blocking=True)
            tgt = tgt.to('cuda', non_blocking=True)

            start_tokens = torch.tensor([tkn('DECSTART')]).to('cuda')
            sample = model.generate(src, start_tokens, DEC_SEQ_LEN, eos_token=tkn('EOS'), mask=src_mask)

            sp = spm.SentencePieceProcessor(model_file=sp_model_path)
            print_stmt = ""
            print_stmt += f"\ninput tokenized:  \n{detokenize_bytes(sp, src.tolist()[0])} \n"
            print_stmt += f"\npredicted detokenized:  \n{detokenize_bytes(sp, sample.tolist())}\n"
            print_stmt += f"\nactual detokenized:     \n{detokenize_bytes(sp, tgt.tolist()[0])}\n"
            print(print_stmt)

    # Clean up
    if use_full_async:
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