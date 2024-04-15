import io
import os
import subprocess
import sys
import time
#import tempfile
#from filelock import FileLock
#import pty

MAIN_PROGRAM = 'sample_tqdm.py'

args = [
    sys.executable, MAIN_PROGRAM,
]


def process():
    #tmp = tempfile.SpooledTemporaryFile()
    #tmp = io.StringIO()
    #master_fd, slave_fd = pty.openpty()
    master_fd, slave_fd = os.openpty()

    with subprocess.Popen(args,
                          env=os.environ,
                          stdin=slave_fd,
                          stdout=slave_fd,
                          stderr=slave_fd,
                          bufsize=0,
                          ) as proc:
        os.close(slave_fd)
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

            line = os.read(master_fd, 100)
            #os.tel
            #os.lseek()

            # write a bunch of newlines or 0's
            #tmp.write(b'\0' * 128)

            #print(f'At {tmp.tell()}')

            #print('here')
            #print(line)

            # yield to other threads
            #time.sleep(0)

            time.sleep(0.25)


if __name__ == '__main__':
    print(process())
