import fcntl
import os
import subprocess
import sys
import time

#BARBERSHOP_DIR = 'C:/Users/rico/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main'
BARBERSHOP_DIR = 'C:/Users/tobin/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main'
BARBERSHOP_MAIN = os.path.join(BARBERSHOP_DIR, 'main.py')

args = [
    sys.executable, BARBERSHOP_MAIN,
    "--im_path1", '0.png',  # face
    "--im_path2", '1.png',  # style
    "--im_path3", '2.png',  # color
    "--sign", 'realistic',
    "--smooth", '5',
]

#master_fd, slave_fd = pty.openpty()

#w = tempfile.NamedTemporaryFile()
def process():
    with subprocess.Popen(args,
                          env=os.environ,
                          cwd=BARBERSHOP_DIR,
                          #stdout=w,
                          stdout=subprocess.PIPE,
                          #stderr=subprocess.STDOUT,
                          stderr=subprocess.PIPE,

                          bufsize=0,
                          #stdout=slave_fd,
                          #stdin=slave_fd,  # redundant/unused?
                          #stderr=slave_fd,
                          ) as barber_proc:
        #os.close(slave_fd)

        fcntl.fcntl(barber_proc.stdout, fcntl.F_SETFL, os.O_NONBLOCK)

        while True:
            result = barber_proc.poll()
            if result == 0:
                return True
            elif result is not None:
                return False

            #barber_proc.stdout.flush()
            #barber_proc.stdout.

            line = barber_proc.stdout.readline()

            #with open(master_fd, 'r') as stdout:
                #for line in stdout:

            print(line)

            # yield to other threads
            time.sleep(0)


if __name__ == '__main__':
    print(process())
