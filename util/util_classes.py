import os
import numpy as np
import time
import shlex
import subprocess
import sys

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


# This is the class for writing both in screen and logfile
# https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()
            
# https://stackoverflow.com/questions/1556348/python-run-a-process-with-timeout-and-capture-stdout-stderr-and-exit-status
class ShellRunner(object):
    """ Runs shell commmand from python """
    def run(self, command, timeout=10):
        proc = subprocess.Popen(command, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        poll_seconds = .250
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() == None:
            time.sleep(poll_seconds)

        if proc.poll() == None:
            if float(sys.version[:3]) >= 2.6:
                proc.terminate()
            raise Exception('Timed Out')

        stdout, stderr = proc.communicate()
        return stdout, stderr, proc.returncode