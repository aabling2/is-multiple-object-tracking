import sys
from os.path import dirname, realpath

# Trabalha no diretotio do pacote
this_path = dirname(realpath(__file__))

# Inclui projeto no python path
pathsrc = dirname(this_path)
if pathsrc not in sys.path:
    sys.path.append(pathsrc)
