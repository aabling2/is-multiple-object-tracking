import sys
from os import chdir
from os.path import dirname, realpath

# Trabalha no diretotio do pacote
this_path = dirname(realpath(__file__))
chdir(this_path)

# Inclui projeto no python path
pathsrc = dirname(this_path)
if pathsrc not in sys.path:
    sys.path.append(pathsrc)