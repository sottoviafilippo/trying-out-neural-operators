# Trying Neural Operators

- `FFEM_building_block.py` contains the definitions of the finite element method in two dimensions.
- `deepOnet_definitions.py` contains the definitions of two DeepONet classes, with and without Fourier encoding.
- `deepOnet_trials.ipynb` is a notebook with examples and tests of the DeepONet classes defined in `deepOnet_definitions.py`. Using as a test solutions of the 2d Poisson equation, the reference solutions being computed with the 2d finite element method found in `FFEM_building_block.py`.
- `create_training_dataset.py` is a code that creates and save a training dataset for the DeepONet classes (with solutions of the 2d Poisson equations as a function of the source term).
- `FNO_definitions.py` contains the definitions of Fourier Neural Operators. Work in progress.
- `FNO_trials.ipynb` tests and trials of Fourier Neural Operators. Just started, work in progress. Following [this review](https://arxiv.org/abs/2512.01421v2).
- `create_training_dataset_for_FNO.py` creates and saves a training dataset for the FNO classes (with solutions of the 2d Poisson equations as a function of the source term).
- `create_dataset_FNO_neuralop.py` creates and saves a training dataset to be using when training an FNO from the neuraloperator library. (https://github.com/neuraloperator/neuraloperator)

Sofar DeepONets give satisfying results, while the FNO that I coded is slow and does not really produce satisfying results. Work in progress. 