# AstroNeRF

Code for artificial star cluster dataset generation using ArtPop and planar image alignment experiment.
This code is largely reworked from Lin et al (https://github.com/chenhsuanlin/bundle-adjusting-NeRF). We specifically worked off their existing planar alignment experiment and adapted it to take in multiple images.

--------------------------------------

### Prerequisites

This code is developed with Python3 (`python3`). PyTorch 1.9+ is required.  
It is recommended use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Install the dependencies and activate the environment `astronerf` with
```bash
conda env create --file requirements.yaml python=3
conda activate astronerf
```
Initialize the external submodule dependencies with
```bash
git submodule update --init --recursive
```

--------------------------------------

### Running the code

- #### ArtPop dataset generation

The options file artpop.yaml contains all of the ArtPop options. For further detail on appropriate values for each option, see ArtPop documentation: https://artpop.readthedocs.io/en/latest/). Additionally, you can decide on the number of images you want to generate, as well as the image size. To generate the dataset, run:

```bash
python3 generatedata.py --yaml=artpop
```

- #### Planar image alignment experiment

The options file barfplanar.yaml contains all of the options for the planar alignment experiment. To train the planar alignment experiment, run:

```bash
python3 train.py --yaml=barfplanar
```

- #### Visualizing the results

--------------------------------------

### Codebase structure
To override specific options, add `--<key>=value` or `--<key1>.<key2>=value` (and so on) to the command line. The configuration will be loaded as the variable `opt` throughout the codebase.  
  
Some tips on using and understanding the codebase:
- The computation graph for forward/backprop is stored in `var` throughout the codebase.
- The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
- If you are using a multi-GPU machine, you can add `--gpu=<gpu_number>` to specify which GPU to use. Multi-GPU training/evaluation is currently not supported.
- To resume from a previous checkpoint, add `--resume=<ITER_NUMBER>`, or just `--resume` to resume from the latest checkpoint.
