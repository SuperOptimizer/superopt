from subprocess import PIPE, run
import os
import platform
import gzip
import tarfile
import tempfile
import shutil
import threading
import time
import concurrent.futures
import multiprocessing
import io

from impl import ROOTDIR, TMP, HOMEDIR

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
        if x % 50 == 0:
            print(f"{threadnum}: {x}")
        ret = run(f'{yarpgen} --std=c++ -o {outdir}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if ret.returncode != 0:
            raise Exception(f"yarpgen failed: {ret.stderr.decode()}")

        ret = run(f'clang++ -c {c_file} -o {unopt_obj} -O0 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if ret.returncode != 0:
            raise Exception(f"clang++ unopt failed: {ret.stderr.decode()}")

        ret = run(f'clang++ -c {c_file} -o {opt_obj} -O3 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if ret.returncode != 0:
            raise Exception(f"clang++ opt failed: {ret.stderr.decode()}")

        ret = run(f'objcopy --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if ret.returncode != 0:
            raise Exception(f"objcopy unopt failed: {ret.stderr.decode()}")

        ret = run(f'objcopy --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if ret.returncode != 0:
            raise Exception(f"objcopy opt failed: {ret.stderr.decode()}")

        with open(unopt_obj, 'rb') as f, open(opt_obj, 'rb') as g:
            ret = (f.read(), g.read())
            yield ret



def gen_model_training_data_binary_archives(samples_per_thread=1000):
    num_threads = multiprocessing.cpu_count()

    os.makedirs(f"{HOMEDIR}/superopt_data", exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    file_counter_lock = threading.Lock()
    file_counter = [len([f for f in os.listdir(f"{HOMEDIR}/superopt_data") if f.endswith('.tar.gz')])]

    def get_next_file_number():
        with file_counter_lock:
            num = file_counter[0]
            file_counter[0] += 1
            return num

    def generate_thread_data(thread_id):
        file_num = get_next_file_number()

        final_archive = f"{HOMEDIR}/superopt_data/corpus_{file_num}.tar.gz"

        print(f"Thread {thread_id}: Generating {samples_per_thread} binary samples for corpus_{file_num}.tar.gz")

        try:
            with tarfile.open(final_archive, 'w:gz') as tar:
                for i, pair in enumerate(gen_yarpgen(thread_id, samples_per_thread)):
                    unopt_bytes, opt_bytes = pair

                    encoder_data = io.BytesIO(unopt_bytes)
                    encoder_info = tarfile.TarInfo(name=f"encoder_{i}.o")
                    encoder_info.size = len(unopt_bytes)
                    tar.addfile(encoder_info, encoder_data)

                    decoder_data = io.BytesIO(opt_bytes)
                    decoder_info = tarfile.TarInfo(name=f"decoder_{i}.o")
                    decoder_info.size = len(opt_bytes)
                    tar.addfile(decoder_info, decoder_data)

                    if (i + 1) % 100 == 0:
                        print(f"Thread {thread_id}: Generated {i + 1}/{samples_per_thread} binary samples")

            print(f"Thread {thread_id}: Completed {samples_per_thread} binary samples -> {final_archive}")
            return final_archive

        except Exception as e:
            print(f"Thread {thread_id}: Error occurred: {e}")
            if os.path.exists(final_archive):
                os.remove(final_archive)
                print(f"Thread {thread_id}: Cleaned up {final_archive}")
            raise

    print(f"Starting parallel binary archive generation with {num_threads} threads, {samples_per_thread} samples each...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(generate_thread_data, i) for i in range(num_threads)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    print(f"Binary archive generation complete! Created {len(results)} corpus archives.")
    return results


def generate_multiple_binary_batches(num_batches=200, samples_per_thread=1000):

    print(f"Generating {num_batches} batches with {samples_per_thread} binary samples per thread...")

    for batch in range(num_batches):
        print(f"\n=== Starting binary batch {batch + 1}/{num_batches} ===")
        gen_model_training_data_binary_archives(samples_per_thread)
        print(f"=== Completed binary batch {batch + 1}/{num_batches} ===")

    print(f"\nAll {num_batches} binary batches completed!")

if __name__ == '__main__':
    generate_multiple_binary_batches(num_batches=20000, samples_per_thread=1000)
