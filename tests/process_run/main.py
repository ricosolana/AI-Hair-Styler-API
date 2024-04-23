import os
import subprocess
import sys

import sys
print(sys.executable)

#BARBER_ALIGN = 'C:/Users/rico/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/align_face.py'
BARBER_ALIGN = 'C:/Users/tobin/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/align_face.py'

process_method = 'system'

if process_method == 'system':
    os.system(f'python {BARBER_ALIGN} '
              #f'-unprocessed_dir C:/Users/rico/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/unprocessed'
              f'-unprocessed_dir C:/Users/tobin/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/unprocessed'
              f'-output_dir ./output')

elif process_method == 'system2':
    cmd32 = os.path.join(os.environ['SYSTEMROOT'], 'SysWOW64', 'cmd.exe')
    subprocess.call('{} /c set SYSTEMROOT'.format(cmd32), env=os.environ)

elif process_method == 'subprocess':
    align_proc = subprocess.run([
        sys.executable, BARBER_ALIGN,
        #'python', BARBER_ALIGN,
        #'C:/Users/rico/AppData/Local/Programs/Python/Python310/python.exe', BARBER_ALIGN,
        '-unprocessed_dir', '../../barber_faces_unprocessed_input/c23f95d3fef44f40a3a7dd8a093802973c445c01f12253fedc26b691cc3fabff',
        "-output_dir", './output'
    ],
        env=os.environ,
        shell=True,
        #cwd='C:/Users/rico/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/')
        cwd='C:/Users/tobin/Documents/GitHub/BarbershopAI-Fork-3/Barbershop-main-20240402T181303Z-001/Barbershop-main/')

    exitcode = align_proc.returncode

    if exitcode != 0:
        print('fail')
    else:
        print('success')

elif process_method == 'subprocess2':
    # C:/Users/rico/AppData/Local/Programs/Python/Python310/python.exe
    #subprocess.check_call(['C:/Users/rico/AppData/Local/Programs/Python/Python310/python.exe', '-c', 'print("Hello World")'])
    #subprocess.check_call([r'C:\Python34_x32\python.exe', '-c', 'print("Hello World")'])
    subprocess.check_call(
        #['cmd.exe', 'C:/Users/rico/AppData/Local/Programs/Python/Python310/python.exe', BARBER_ALIGN])
        ['cmd.exe', 'C:/Users/tobin/AppData/Local/Programs/Python/Python310/python.exe', BARBER_ALIGN])

elif process_method == 'win32':
    print('NYI')
    pass

print('end')
