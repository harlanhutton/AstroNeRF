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

### Dataset

--------------------------------------

### Running the code

- #### ArtPop dataset generation

- #### Planar image alignment experiment

- #### Visualizing the results

--------------------------------------

### Codebase structure

The main engine and network architecture in `model/barf.py` inherit those from `model/nerf.py`.
This codebase is structured so that it is easy to understand the actual parts BARF is extending from NeRF.
It is also simple to build your exciting applications upon either BARF or NeRF -- just inherit them again!
This is the same for dataset files (e.g. `data/blender.py`).

To understand the config and command lines, take the below command as an example:
```bash
python3 train.py --group=<GROUP> --model=barf --yaml=barf_blender --name=<NAME> --data.scene=<SCENE> --barf_c2f=[0.1,0.5]
```
This will run `model/barf.py` as the main engine with `options/barf_blender.yaml` as the main config file.
Note that `barf` hierarchically inherits `nerf` (which inherits `base`), making the codebase customizable.  
The complete configuration will be printed upon execution.
To override specific options, add `--<key>=value` or `--<key1>.<key2>=value` (and so on) to the command line. The configuration will be loaded as the variable `opt` throughout the codebase.  
  
Some tips on using and understanding the codebase:
- The computation graph for forward/backprop is stored in `var` throughout the codebase.
- The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
- If you are using a multi-GPU machine, you can add `--gpu=<gpu_number>` to specify which GPU to use. Multi-GPU training/evaluation is currently not supported.
- To resume from a previous checkpoint, add `--resume=<ITER_NUMBER>`, or just `--resume` to resume from the latest checkpoint.
- (to be continued....)
