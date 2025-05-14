from  subprocess import PIPE, run
import os
import platform
import gzip




from impl import ROOTDIR, TMP, bytes_to_hex_string

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
  os.makedirs(TMP, exist_ok=True)
  progs = gen_yarpgen(0,2000)

  encoder_corpus = f"{TMP}/encoder.txt"
  decoder_corpus = f"{TMP}/decoder.txt"

  with open(encoder_corpus, 'at') as f, open(decoder_corpus, 'at') as g:
    for pair in progs:
      unopt,opt = pair
      f.write(bytes_to_hex_string(unopt) + "\n")
      g.write(bytes_to_hex_string(opt) + "\n")
  #spm_train --input=encoder.txt --model_prefix=encoder --vocab_size=4096 --max_sentence_length=655350 --character_coverage=1.0 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false
  #run(f"spm_train --input={encoder_corpus} --model_prefix=encoder --vocab_size=8192 --character_coverage=1.0 --model_type=unigram --max_sentence_length=65535 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false".split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,cwd=TMP)
  #run(f"spm_train --input={decoder_corpus} --model_prefix=decoder --vocab_size=8192 --character_coverage=1.0 --model_type=unigram --max_sentence_length=65535 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false".split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,cwd=TMP)


def gen_model_training_data_parallel():
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 20000

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
  encoder_corpus = f"{TMP}/encoder.txt.gzip"
  decoder_corpus = f"{TMP}/decoder.txt.gzip"

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