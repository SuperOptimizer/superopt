from  subprocess import PIPE, run
import os
import platform
import gzip




from impl import ROOTDIR, TMP, bytes_to_hex_string,  NUM_VOCAB_TOKENS

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 '

GCC = 'gcc'
CLANG = 'clang'
CLANGPP = 'clang++'
STRIP = 'strip'
OBJDUMP = 'objdump'
OBJCOPY = 'objcopy'


def gen_yarpgen(threadnum, num):
  yarpgen = f'/{ROOTDIR}/bin/{platform.system()}/yarpgen'
  outdir = f'/{TMP}/yarpgen_{threadnum}'
  os.makedirs(outdir, exist_ok=True)
  c_file = f'{outdir}/func.c'
  opt_obj = f'{outdir}/func.opt.o'
  unopt_obj = f'{outdir}/func.unopt.o'

  for x in range(num):
    print(x)
    ret = run(f'{yarpgen} --std=c -o {outdir}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang -c {c_file} -o {unopt_obj} -include stdint.h -O0 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang -c {c_file} -o {opt_obj}   -include stdint.h -O3 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'objcopy  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'objcopy  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    with open(unopt_obj, 'rb') as f, open(opt_obj, 'rb') as g:
      ret=(f.read(),g.read())
      yield ret

def gen_model_training_data():
  os.makedirs(TMP, exist_ok=True)
  progs = gen_yarpgen(0, 2000)

  encoder_corpus = f"{TMP}/encoder.txt.gzip"
  decoder_corpus = f"{TMP}/decoder.txt.gzip"

  with gzip.open(encoder_corpus, 'at') as f, gzip.open(decoder_corpus, 'at') as g:
    for pair in progs:
      unopt, opt = pair
      f.write(bytes_to_hex_string(unopt) + "\n")
      g.write(bytes_to_hex_string(opt) + "\n")

def gen_sentencepiece_training_data():
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 50000

  # Split workload across threads
  programs_per_thread = total_programs // num_threads
  remainder = total_programs % num_threads

  os.makedirs(TMP, exist_ok=True)

  print(f"Starting parallel generation with {num_threads} threads...")

  # Function to generate programs for a specific thread
  def generate_for_thread(thread_id):
    num_programs = programs_per_thread + (1 if thread_id < remainder else 0)
    temp_encoder_file = f"{TMP}/encoder_sp_{thread_id}.txt"
    temp_decoder_file = f"{TMP}/decoder_sp_{thread_id}.txt"

    with open(temp_encoder_file, 'wt') as f, open(temp_decoder_file, 'wt') as g:
      for pair in gen_yarpgen(thread_id, num_programs):
        unopt, opt = pair
        f.write(bytes_to_hex_string(unopt) + "\n")
        g.write(bytes_to_hex_string(opt) + "\n")

    return (temp_encoder_file, temp_decoder_file)

  # Create output files
  encoder_corpus = f"{TMP}/encoder.txt"
  decoder_corpus = f"{TMP}/decoder.txt"

  # Run tasks in parallel using threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_for_thread, i) for i in range(num_threads)]
    temp_files = [future.result() for future in concurrent.futures.as_completed(futures)]

  print("Data generation complete. Combining files...")

  # Combine all temporary files into final output
  with open(encoder_corpus, 'at') as f_enc, open(decoder_corpus, 'at') as f_dec:
    for enc_file, dec_file in temp_files:
      with open(enc_file, 'rt') as temp_enc, open(dec_file, 'rt') as temp_dec:
        f_enc.write(temp_enc.read())
        f_dec.write(temp_dec.read())

      # Cleanup temporary files
      os.remove(enc_file)
      os.remove(dec_file)

  print("File combination complete. Training SentencePiece models...")

  # Train SentencePiece models with unigram
  print("Training encoder model...")
  ret = run(f"spm_train --input={encoder_corpus} "
            f"--model_prefix={TMP}/encoder "
            f"--vocab_size={NUM_VOCAB_TOKENS} "
            f"--character_coverage=1.0 "
            f"--model_type=unigram "
            f"--max_sentence_length=65535 "
            f"--bos_id=-1 --eos_id=-1 --pad_id=-1 "
            f"--max_sentencepiece_length=32 "
            f"--num_threads=32 "
            f"--add_dummy_prefix=false "
            f"--train_extremely_large_corpus=true "
            f"--split_by_number=false".split(),
            stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=TMP)

  print(ret.stdout.decode('utf-8'))
  if ret.stderr:
    print(f"Encoder stderr: {ret.stderr.decode('utf-8')}")

  print("Training decoder model...")
  ret = run(f"spm_train --input={decoder_corpus} "
            f"--model_prefix={TMP}/decoder "
            f"--vocab_size={NUM_VOCAB_TOKENS} "
            f"--character_coverage=1.0 "
            f"--model_type=unigram "
            f"--max_sentence_length=65535 "
            f"--bos_id=-1 --eos_id=-1 --pad_id=-1 "
            f"--max_sentencepiece_length=32 "
            f"--num_threads=32 "
            f"--add_dummy_prefix=false "
            f"--train_extremely_large_corpus=true "
            f"--split_by_number=false".split(),
            stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=TMP)

  print(ret.stdout.decode('utf-8'))
  if ret.stderr:
    print(f"Decoder stderr: {ret.stderr.decode('utf-8')}")

  print("SentencePiece training complete!")

def gen_model_training_data_parallel(gzip_num):
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 2000

  # Split workload across threads
  programs_per_thread = total_programs // num_threads
  remainder = total_programs % num_threads

  os.makedirs(TMP, exist_ok=True)

  # Function to generate programs for a specific thread
  def generate_for_thread(thread_id):
    num_programs = programs_per_thread + (1 if thread_id < remainder else 0)
    temp_encoder_file = f"{TMP}/encoder_{thread_id}.txt.gzip"
    temp_decoder_file = f"{TMP}/decoder_{thread_id}.txt.gzip"

    with gzip.open(temp_encoder_file, 'wt') as f, gzip.open(temp_decoder_file, 'wt') as g:
      for pair in gen_yarpgen(thread_id, num_programs):
        unopt, opt = pair
        f.write(bytes_to_hex_string(unopt) + "\n")
        g.write(bytes_to_hex_string(opt) + "\n")

    return (temp_encoder_file, temp_decoder_file)

  # Create output files
  encoder_corpus = f"{TMP}/encoder_corpus_{gzip_num}.txt.gzip"
  decoder_corpus = f"{TMP}/decoder_corpus_{gzip_num}.txt.gzip"

  # Run tasks in parallel using threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_for_thread, i) for i in range(num_threads)]
    temp_files = [future.result() for future in concurrent.futures.as_completed(futures)]

  # Combine all temporary files into final output
  with gzip.open(encoder_corpus, 'at') as f_enc, gzip.open(decoder_corpus, 'at') as f_dec:
    for enc_file, dec_file in temp_files:
      with gzip.open(enc_file, 'rt') as temp_enc, gzip.open(dec_file, 'rt') as temp_dec:
        f_enc.write(temp_enc.read())
        f_dec.write(temp_dec.read())

      # Cleanup temporary files
      os.remove(enc_file)
      os.remove(dec_file)

if __name__ == '__main__':
    for i in range(20):
      gen_model_training_data_parallel(i)
    #gen_sentencepiece_training_data()
    #gen_model_training_data()