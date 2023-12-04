from sys import argv
from diffuse.main import main

if __name__ == '__main__':
    main(argv[0], '--config', 'reale-run.cfg', '--load-model', './RealeRun')
