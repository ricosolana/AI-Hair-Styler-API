import os
import subprocess
import sys
import time
import winpty

MAIN_PROGRAM = 'sample_tqdm.py'

args = [
    sys.executable, MAIN_PROGRAM,
]

def process():
    with subprocess.Popen(args,
                          env=os.environ,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          bufsize=0,
                          ) as proc:
        while True:
            result = proc.poll()
            if result == 0:
                return True
            elif result is not None:
                return False

            # doesnt read when tqdm changes buffer
            line = proc.stdout.readline()

            print(line)

            # yield to other threads
            time.sleep(0)


if __name__ == '__main__':
    print(process())
