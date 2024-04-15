import io
import os
import subprocess
import sys
import time
import tempfile
import numpy as np
#import pty # test this now

MAIN_PROGRAM = 'sample_tqdm.py'

__nums = [2, 4, 5, 3]

args = [
    sys.executable, MAIN_PROGRAM,
]


def process():
    tmp = tempfile.SpooledTemporaryFile(max_size=2048)
    #tmp = io.StringIO()

    last = time.perf_counter()

    with subprocess.Popen(args,
                          env=os.environ,
                          stdout=tmp,
                          stderr=subprocess.STDOUT,
                          bufsize=0,
                          ) as proc:
        while True:
            result = proc.poll()
            if result == 0:
                return True
            elif result is not None:
                return False

            # doesnt read when tqdm changes buffer
            #proc.stdout.write(b'\r')
            #proc.stdout.flush()
            #line = proc.stdout.readline()

            # read
            #line = tmp.readline()
            #tmp.seek(0, io.SEEK_SET)
            #print(line)

            # write a bunch of newlines or 0's
            #tmp.write(b'\0' * 128)

            #print(f'At {tmp.tell()}')

            now = time.perf_counter()
            if now - last > 0.25:
                last = now
                # whether file position moves in this short time
                tmp.seek(0, io.SEEK_SET)
                print(tmp.readline())
                #pos = tmp.seek(0, io.SEEK_END)
                pos = tmp.tell()
                print(f'Size {pos}')

            tmp.seek(0, io.SEEK_SET)

            #print('here')
            #print(line)

            # yield to other threads
            #time.sleep(0)

            time.sleep(0.01)
            #time.sleep(0)


if __name__ == '__main__':
    print(process())
