import os
import sys

sys.path.extend(os.path.abspath('..'))
os.environ['PYTHONPATH'] = os.path.abspath('..')
print(os.environ['PYTHONPATH'])
from data import trainGenerator
