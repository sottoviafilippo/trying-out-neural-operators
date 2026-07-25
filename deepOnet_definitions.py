import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable


class deepOnet_Poisson:
    """Defines and sets up a deepOnet"""
    # (for the moment) the sampling points of the input functions are fixed

    def __init__(self, function_discretization_size:int, N_nodes_branch:int, N_nodes_trunk:int, latent_dimension:int, domain_dimension:int = 2):

        self.N_nodes_branch = N_nodes_branch
        self.N_nodes_trunk  = N_nodes_trunk
        self.function_discretization_size = function_discretization_size # number of points at which the input function is evaluated BEFORE feeding it into the dOn
        self.latent_dimension = latent_dimension #size of the output of the trunk and branch network
        self.domain_dimension = domain_dimension # usually 2, since for the moment I am trying to solve the Poisson equation in 2d

        # don't use ReLU for physics-informed networks because second derivatives vanish

        # for the moment I am using two hidden layers. To be corrected later if necessary
        self.trunk_network = nn.Sequential(
            nn.Linear(self.function_discretization_size, self.N_nodes_branch),  
            nn.ReLU(),         
            nn.Linear(self.N_nodes_branch, self.N_nodes_branch),  
            nn.ReLU(),   
            nn.Linear(self.N_nodes_branch, self.latent_dimension)
        )

        self.trunk_network = nn.Sequential(
            nn.Linear(self.domain_dimension, self.N_nodes_trunk),  
            nn.Tanh(),         
            nn.Linear(self.N_nodes_trunk, self.N_nodes_trunk),  
            nn.Tanh(),   
            nn.Linear(self.N_nodes_trunk, self.latent_dimension)
        )

        self.optimizer_branch = optim.Adam(self.branch_network.parameters(), lr=0.01)
        self.optimizer_trunk  = optim.Adam(self.trunk_network.parameters(), lr=0.01)

        pass