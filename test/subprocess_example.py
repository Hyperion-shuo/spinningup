import subprocess
import sys
# from this blog
# https://www.cnblogs.com/lincappu/p/8270709.html


# seems little differences
# returncode = subprocess.call(['ls', '-l'])
# returncode = subprocess.check_call(['ls', '-l'])
returncode = subprocess.check_output(['ls', '-l'])
print(returncode)