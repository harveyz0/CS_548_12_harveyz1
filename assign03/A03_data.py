from sys import argv
from diffuse.main import main

if __name__ == '__main__':
    main(argv[0], '--resize', '--config', 'reale-run.cfg')
