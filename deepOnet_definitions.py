import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# note: many comments are pedagogical and pretty basic: written for myself while writing this class, for learning

class deepOnet_Poisson(nn.Module):
    """Defines and sets up a deepOnet, trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed

    def __init__(self, function_discretization_size:int, N_nodes_branch:int, N_nodes_trunk:int, latent_dimension:int, domain_dimension:int = 2):

        super().__init__() # refers to nn.Module

        self.N_nodes_branch = N_nodes_branch
        self.N_nodes_trunk  = N_nodes_trunk
        self.function_discretization_size = function_discretization_size # number of points at which the input function is evaluated BEFORE feeding it into the dOn
        self.latent_dimension = latent_dimension #size of the output of the trunk and branch network
        self.domain_dimension = domain_dimension # usually 2, since for the moment I am trying to solve the Poisson equation in 2d

        # don't use ReLU for physics-informed networks because second derivatives vanish

        # for the moment I am using two hidden layers. To be corrected later if necessary
        self.branch_network = nn.Sequential(
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

        # If I had not subclassed nn.Module .parameters() would not be defined
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()


    def forward(self, branch_input, trunk_input):
        """output of the model: scalar product between the outputs of the trunk and branch network"""

        branch_output = self.branch_network(branch_input)   
        trunk_output  = self.trunk_network(trunk_input)      
        # outer product over the two batch dims, contracted on latent dimension

        return torch.einsum("bl,tl->bt", branch_output, trunk_output)


    def fit(self, num_epochs:int, branch_X, trunk_X, Y, print_progress = True):
        # X, Y in the training dataset. 
        
        self.epochs = []
        self.losses = []

        # First do some checks. Dimensions have to match, in particular the size of the array representing the input function itself
        # First check the branch network dimension
        if branch_X.shape[1] != self.function_discretization_size:
            raise ValueError(
                f"Branch input mismatch! The model expects functions discretized at " f"{self.function_discretization_size} points, but got {branch_X.shape[1]}."
            )
            
        # Then check the trunk network input dimension (2 for the 2d Poisson equation, which I am taking as first test case)
        if trunk_X.shape[1] != self.domain_dimension:
            raise ValueError(
                f"Trunk input mismatch! The model expects a domain dimension of " f"{self.domain_dimension}, but got {trunk_X.shape[1]}."
            )

        for epoch in range(num_epochs):
    
            predictions = self(branch_X, trunk_X)  # __call__ "does some bookkeeping" and calls the previously defined self.forward()
            loss = self.criterion(predictions, Y)

            self.optimizer.zero_grad() # otherwise the gradients would be added to the previously computed ones
            loss.backward() # computes the gradient of the loss compute on the neural network, via model(X)
            self.optimizer.step() # updates the weights

            # Track the loss history
            self.epochs.append(epoch + 1)
            self.losses.append(loss.item())

            if print_progress and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    def fit_from_npz(self, training_dataset:str, num_epochs:int, print_progress = True):

        """Fits the model using a training dataset stored in a .npz file"""
        training_data = np.load(training_dataset)

        coords = training_data["coords"]
        training_branch = training_data["f"]
        training_Y = training_data["u"]

        self.fit(num_epochs, training_branch, coords, training_Y, print_progress = print_progress)

        pass



# to do: implement train/eval split to prevent overfitting
# to do: implement GPU usage
# to do: mini-batching