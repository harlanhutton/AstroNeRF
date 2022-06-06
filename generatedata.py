import importlib
import os,sys
import torch
import importlib
import options
from util import log

def main():

    log.process(os.getpid())
    log.title("[{}] (ArtPop code for Generating Images)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    ap = importlib.import_module("run_artpop")
    a = ap.ArtPopGenerator(opt)
    #.source(opt)
    a.save(opt)


if __name__=="__main__":
    main()