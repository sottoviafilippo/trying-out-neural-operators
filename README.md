# Trying Neural Operators

- `FFEM_building_block.py` contains the definitions of the finite element method in two dimensions.
- `deepOnet_definitions.py` contains the definitions of two DeepONet classes, with and without Fourier encoding.
- `deepOnet_trials.ipynb` is a notebook with examples and tests of the DeepONet classes defined in `deepOnet_definitions.py`. Using as a test solutions of the 2d Poisson equation, the reference solutions being computed with the 2d finite element method found in `FFEM_building_block.py`. The DeepONet class is called by `deepOnet_fourier_embedded_v2(x_coords_for_branch, N_nodes_branch, N_nodes_trunk, latent_dimension)` (`N_nodes` being the size of the hidden layers in the corresponding parts of the DeepONet). A second class, `deepOnet_fourier_embedded_v2`, uses Fourier encoding. For the moment the number of layers in the network cannot be set by the user, it is hardcoded (two hidden layers for the trunk network and three hidden layers for the branch network).
- `create_training_dataset.py` is a code that creates and save a training dataset for the DeepONet classes (with solutions of the 2d Poisson equations as a function of the source term).
- `FNO_definitions.py` contains the definitions of Fourier Neural Operators. Work in progress.
- `FNO_trials.ipynb` tests and trials of Fourier Neural Operators. Work in progress. Following [this review](https://arxiv.org/abs/2512.01421v2) for the theory and algorithms. Also trying FNO models from the neuraloperator library (https://github.com/neuraloperator/neuraloperator). The FNO class is called by `FNO_v1(n_modes, hidden_dimension, lr)`, where `n_modes` is a list containing the number of Fourier modes to be retained in all directions. 
- `create_training_dataset_for_FNO.py` creates and saves a training dataset for the FNO classes (with solutions of the 2d Poisson equations as a function of the source term).
- `create_dataset_FNO_neuralop.py` creates and saves a training dataset to be using when training an FNO from the neuraloperator library. 

Results, sofar: 
- DeepONets give satisfying results as one can see in the plots in the `deepOnet_trials.ipynb` notebook.
- the FNO that I coded gives, at best, mediocre results (and so does the model from the neuraloperator library). But sofar I tested both models with only 30 epochs (for a first quick check, preventing overfitting), so work is still in progress. The results obtained with `neuraloperator` do not seem to be much better anyways. Probably (also) an issue concerning the small size of the training dataset.