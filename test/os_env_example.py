# Python program to explain os.getenv() method

# importing os module
import os
import sys

key = 'HOME'
value = os.getenv(key)
print("Value of 'HOME' environment variable :", value)

key = 'JAVA_HOME'
value = os.getenv(key)
print("Value of 'JAVA_HOME' environment variable :", value)

key="IN_MPI"
value = os.getenv(key, "value does not exist")
print("Value of 'IN_MPI' environment variable :", value)

os.environ.update(
    IN_MPI="1"
)
print("Value of 'IN_MPI' environment variable :", os.getenv('IN_MPI'))

# current python.exe path
print('sys.excutable return current python.exe path: ', sys.executable)
# current python lib path
print('sys.prefix return current python lib path: ', sys.prefix)
# return file name and args when calling using python
# try run python os_env_example.py we are args
print(sys.argv)


# in macos seems no difference, which are all dicts
# print(os.environ.keys())
print(os.environ)