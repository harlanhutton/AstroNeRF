import importlib
import os,sys
import torch
import importlib
import options
from util import log, uniquify
import warnings

def main():

    warnings.filterwarnings("ignore", category=UserWarning)

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    if opt.device == 'cuda':
        with torch.cuda.device(opt.device):
            model = importlib.import_module("barfplanar")
            file = model.File(opt)
            m = model.Model(opt)
            m.load_data(opt)
            m.build_networks(opt)
            m.setup_optimizer(opt)
            m.train(opt,file)
            m.predict_entire_image(opt,file)
    else:
        model = importlib.import_module("barfplanar")
        file = model.File(opt)
        m = model.Model(opt)
        m.load_data(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.train(opt,file)
        m.predict_entire_image(opt,file)


if __name__=="__main__":
    main()