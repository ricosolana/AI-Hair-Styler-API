import sys
import time
from winpty import PtyProcess

MAIN_PROGRAM = 'sample_tqdm.py'

args = [
    sys.executable, MAIN_PROGRAM,
]


def process():
    proc = PtyProcess.spawn(' '.join(args))

    # TODO change to forever loop
    # TODO test subprocess pytohn opened and behaviour
    #   versus child being opened
    while proc.isalive():
        def readlinelike(proc: PtyProcess):
            buf = []
            while 1:
                try:
                    ch = proc.read(1)
                except EOFError:
                    return ''.join(buf)
                buf.append(ch)
                if ch in ('\n', '\r'):
                    return ''.join(buf)

        def readlinelikenoblock(proc: PtyProcess):
            #barber_proc.pty.read(1)
            buf = []
            while 1:
                try:
                    ch = proc.pty.read(1)
                except EOFError:
                    return ''.join(buf)
                buf.append(ch)
                if ch in ('\n', '\r'):
                    return ''.join(buf)



        #print(proc.exitstatus)
        line = readlinelike(proc)
        """
        proc.readline()
        line = leftover + proc.read() # TODO test blocking behaviour when exited

        split = [e for e in line.splitlines(keepends=True) if e.strip()]

        leftover = split[-1][-1]

        #leftover = line.in

        # [e for e in '\rdata\r\n  \n\rafter\n'.splitlines() if e.strip() != '']

        # append
        #if not line[-1] in ('\r', '\n', '\r\n'):

        #buffered_lines += line.split('\r')
        """

        #print(line.encode('utf-8'))
        print(line)
        time.sleep(0.001)

    print(proc.exitstatus)


if __name__ == '__main__':
    print(process())
