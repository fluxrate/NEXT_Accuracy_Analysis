import subprocess as sp
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')


def gpu_count():

    ps_command = sp.Popen("nvidia-smi -L | wc -l", shell=True, stdout=sp.PIPE, stderr=DEVNULL)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    
    return int(ps_output)

    
