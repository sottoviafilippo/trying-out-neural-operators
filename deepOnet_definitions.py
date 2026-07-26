import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# note: many comments are pedagogical and pretty basic: written for myself while writing this class, for learning

class deepOnet_v1(nn.Module):
    """Defines and sets up a deepOnet. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed

    def __init__(self, x_coords_for_branch, N_nodes_branch:int, N_nodes_trunk:int, latent_dimension:int, domain_dimension:int = 2):

        super().__init__() # refers to nn.Module

        self.N_nodes_branch = N_nodes_branch
        self.N_nodes_trunk  = N_nodes_trunk
        self.x_coords_for_branch = x_coords_for_branch
        self.function_discretization_size = x_coords_for_branch.shape[0] # number of points at which the input function is evaluated BEFORE feeding it into the dOn
        self.latent_dimension = latent_dimension #size of the output of the trunk and branch network
        self.domain_dimension = domain_dimension # usually 2, since for the moment I am trying to solve the Poisson equation in 2d
        # the latent dimension is called p in the 2021 paper by Lu Jin and Karniadakis

        # don't use ReLU for physics-informed networks because second derivatives vanish. "in practice p is at least of the order of 10"

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
            nn.Linear(self.N_nodes_trunk, self.N_nodes_trunk),  
            nn.Tanh(),   
            nn.Linear(self.N_nodes_trunk, self.latent_dimension)
        )

        # in Lu Jin Karniadakis they have branch depth 2, trunk depth 3 as standard

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

        coords_branch = training_data["coords_branch"]
        coords_trunk = training_data["coords_trunk"]
        training_branch = training_data["f"]
        training_Y = training_data["u"]

        # this check is needed to ensure future reproducibility of the results
        if coords_branch.any() != self.x_coords_for_branch.any():
                    raise ValueError(
                        f"Branch input mismatch! Function is not discretised on the same points as expected"
                    )


        # before training I need to convert the data to torch objects
        branch_X = torch.from_numpy(training_branch).float()
        trunk_X  = torch.from_numpy(coords_trunk).float()
        Y        = torch.from_numpy(training_Y).float()

        self.fit(num_epochs, branch_X, trunk_X, Y, print_progress=print_progress)

        pass

    def map_function_to_output_at_points(self, f: Callable, points_for_evaluation):

        branch = np.array([f(x[0], x[1]) for x in self.x_coords_for_branch])
        branch = torch.from_numpy(branch).float().unsqueeze(0)
        # unsqueeze(0) turns (N_points,) into (1, N_points)

        return self(branch, torch.from_numpy(points_for_evaluation).float())


# to do: implement train/eval split to prevent overfitting
# to do: implement GPU usage
# to do: mini-batching