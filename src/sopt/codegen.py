from  subprocess import PIPE, run
import os
import platform
import gzip




from impl import ROOTDIR, TMP, bytes_to_hex_string,  NUM_VOCAB_TOKENS, HOMEDIR

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
  c_file = f'{outdir}/func.cpp'
  opt_obj = f'{outdir}/func.opt.o'
  unopt_obj = f'{outdir}/func.unopt.o'

  for x in range(num):
    print(x)
    ret = run(f'{yarpgen} --std=c++ -o {outdir}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang++ -c {c_file} -o {unopt_obj}  -O0 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang++ -c {c_file} -o {opt_obj}    -O3 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
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


def gen_sentencepiece_training_data():
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 10000

  # Split workload across threads
  programs_per_thread = total_programs // num_threads
  remainder = total_programs % num_threads

  os.makedirs(TMP, exist_ok=True)

  print(f"Starting parallel generation with {num_threads} threads...")

  # Function to generate programs for a specific thread
  def generate_for_thread(thread_id):
    num_programs = programs_per_thread + (1 if thread_id < remainder else 0)
    temp_corpus_file = f"{TMP}/corpus_{thread_id}.txt"

    with open(temp_corpus_file, 'wt') as f:
      for pair in gen_yarpgen(thread_id, num_programs):
        unopt, opt = pair
        # Write both unoptimized and optimized to the same corpus for shared vocabulary
        f.write(bytes_to_hex_string(unopt) + "\n")
        f.write(bytes_to_hex_string(opt) + "\n")

    return temp_corpus_file

  # Create output file
  combined_corpus = f"{TMP}/combined_corpus.txt"

  # Run tasks in parallel using threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_for_thread, i) for i in range(num_threads)]
    temp_files = [future.result() for future in concurrent.futures.as_completed(futures)]

  print("Data generation complete. Combining files...")

  # Combine all temporary files into final output
  with open(combined_corpus, 'at') as f_combined:
    for temp_file in temp_files:
      with open(temp_file, 'rt') as temp:
        f_combined.write(temp.read())

      # Cleanup temporary files
      os.remove(temp_file)

  print("File combination complete. Training SentencePiece model...")

  # Train single SentencePiece model with unigram for shared vocabulary
  print("Training combined model...")
  ret = run(f"spm_train --input={combined_corpus} "
            f"--model_prefix={TMP}/superopt "
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
    print(f"Combined model stderr: {ret.stderr.decode('utf-8')}")

  print("SentencePiece training complete!")

def gen_model_training_data_parallel():
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 5000

  # Split workload across threads
  programs_per_thread = total_programs // num_threads
  remainder = total_programs % num_threads

  os.makedirs(TMP, exist_ok=True)
  os.makedirs(f"{HOMEDIR}/superopt_data", exist_ok=True)

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
  gzip_num = len(os.listdir(f"{HOMEDIR}/superopt_data/"))//2
  # Create output files
  encoder_corpus = f"{HOMEDIR}/superopt_data/encoder_corpus_{gzip_num}.txt.gzip"
  decoder_corpus = f"{HOMEDIR}/superopt_data/decoder_corpus_{gzip_num}.txt.gzip"
  print("outputting",encoder_corpus,decoder_corpus)

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
      gen_model_training_data_parallel()
    #gen_sentencepiece_training_data()
    #gen_model_training_data()