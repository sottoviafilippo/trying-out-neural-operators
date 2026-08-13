import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# note: many comments are pedagogical and pretty basic: written for myself while writing this class, for learning


class deepOnet_v1(nn.Module):
    """Defines and sets up a deepOnet. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed
    # note: for the moment I am working on the [-1, 1] square. for general case better to normalize the coordinates

    def __init__(self, x_coords_for_branch, N_nodes_branch:int, N_nodes_trunk:int, latent_dimension:int, domain_dimension:int = 2):

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        super().__init__() # refers to nn.Module

        self.N_nodes_branch = N_nodes_branch
        self.N_nodes_trunk  = N_nodes_trunk
        self.x_coords_for_branch = x_coords_for_branch
        self.function_discretization_size = x_coords_for_branch.shape[0] # number of points at which the input function is evaluated BEFORE feeding it into the dOn
        print("Function discretization size = ", self.function_discretization_size) # = N_x x N_y, check
        self.latent_dimension = latent_dimension #size of the output of the trunk and branch network
        self.domain_dimension = domain_dimension # usually 2, since for the moment I am trying to solve the Poisson equation in 2d
        # the latent dimension is called p in the 2021 paper by Lu Jin and Karniadakis

        # don't use ReLU for physics-informed networks because second derivatives vanish. "in practice p is at least of the order of 10"

        # for the moment I am using two hidden layers. To be corrected later if necessary
        self.branch_network = nn.Sequential(
            nn.Linear(self.function_discretization_size, self.N_nodes_branch),  
            nn.Sigmoid(),        
            nn.Linear(self.N_nodes_branch, self.N_nodes_branch),  
            nn.Sigmoid(),   
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.to(self.device) 

    def forward(self, branch_input, trunk_input):
        """output of the model: scalar product between the outputs of the trunk and branch network"""

        branch_output = self.branch_network(branch_input)   
        trunk_output  = self.trunk_network(trunk_input)      
        # outer product over the two batch dims, contracted on latent dimension

        return torch.einsum("bl,tl->bt", branch_output, trunk_output)


    def fit(self, num_epochs:int, branch_X, trunk_X, Y, branch_X_eval, trunk_X_eval, Y_eval, print_progress = True):
        # X, Y in the training dataset. 

        print("Y std across samples:", Y.std(dim=0).mean().item())
        print("Y std across points:", Y.std(dim=1).mean().item())
        
        self.epochs = []
        self.losses = []
        self.losses_eval = []
        self.rel_errors_eval = []

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

            with torch.no_grad(): # I do not need to track gradients here
                predictions_eval = self(branch_X_eval, trunk_X_eval)
                loss_eval = self.criterion(predictions_eval, Y_eval)
                rel_error_eval = torch.mean((predictions_eval - Y_eval)**2) / torch.mean(Y_eval**2)
            self.losses_eval.append(loss_eval.item())
            self.rel_errors_eval.append(rel_error_eval.item())

            if print_progress and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss eval: {loss_eval.item():.4f}")

                with torch.no_grad(): # check variations - is the model actually learning anything ?
                    preds = self(branch_X, trunk_X)
                    print("prediction std across samples:", preds.std(dim=0).mean().item())
                    print("prediction std across points:", preds.std(dim=1).mean().item())
                    


    def fit_from_npz(self, training_dataset:str, num_epochs:int, eval_rel_size = 0.2, print_progress = True):

        """Fits the model using a training dataset stored in a .npz file"""
        training_data = np.load(training_dataset)

        N_samples = training_data["f"].shape[0]
        N_train = int(N_samples * (1 - eval_rel_size))

        coords_branch = training_data["coords_branch"]
        coords_trunk = training_data["coords_trunk"]
        training_branch = training_data["f"][:N_train]
        training_Y = training_data["u"][:N_train]

        eval_branch = training_data["f"][N_train:]
        eval_Y = training_data["u"][N_train:]

        # this check is to ensure future reproducibility of the results
        if not np.array_equal(coords_branch, self.x_coords_for_branch):
            raise ValueError("Branch input mismatch! Function is not discretised on the same points as expected")

        # before training I need to convert the data to torch objects
        branch_X = torch.from_numpy(training_branch).float().to(self.device)
        trunk_X  = torch.from_numpy(coords_trunk).float().to(self.device)
        Y        = torch.from_numpy(training_Y).float().to(self.device)

        # to device, drectly here
        branch_X_eval = torch.from_numpy(eval_branch).float().to(self.device)
        Y_eval        = torch.from_numpy(eval_Y).float().to(self.device)

        self.fit(num_epochs, branch_X, trunk_X, Y, branch_X_eval = branch_X_eval, trunk_X_eval = trunk_X, Y_eval = Y_eval, print_progress=print_progress)


    def map_function_to_output_at_points(self, f: Callable, points_for_evaluation):

        branch = np.array([f(x[0], x[1]) for x in self.x_coords_for_branch])
        branch = torch.from_numpy(branch).float().unsqueeze(0).to(self.device)
        # unsqueeze(0) turns (N_points,) into (1, N_points)

        return self(branch, torch.from_numpy(points_for_evaluation).float().to(self.device))


class FourierEncoding(nn.Module):
    """Normalizes each coordinate to (0,1) via its own (min, max), then maps
    to (sin, cos) pairs """

    def __init__(self, mins, maxs, epsilon = 0.01):
        super().__init__()
        mins = torch.as_tensor(mins, dtype=torch.float32)
        maxs = torch.as_tensor(maxs, dtype=torch.float32)
        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)
        self.epsilon = epsilon
        
    def forward(self, x):
        x_norm = (x - self.mins) / (self.maxs - self.mins)  # bring to (0,1)
        x_scaled = x_norm * (2. * torch.pi - self.epsilon)
        return torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1) # concatenate output for all input x


class deepOnet_fourier_embedded_v2(nn.Module):
    """Defines and sets up a deepOnet. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed
    # note: for the moment I am working on the [-1, 1] square. for general case better to normalize the coordinates


    def __init__(self, x_coords_for_branch, N_nodes_branch:int, N_nodes_trunk:int, latent_dimension:int, domain_dimension:int = 2, x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0):

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                
        super().__init__() # refers to nn.Module

        self.N_nodes_branch = N_nodes_branch
        self.N_nodes_trunk  = N_nodes_trunk
        self.x_coords_for_branch = x_coords_for_branch
        self.function_discretization_size = x_coords_for_branch.shape[0] # number of points at which the input function is evaluated BEFORE feeding it into the dOn
        print("Function discretization size = ", self.function_discretization_size) # = N_x x N_y, check
        self.latent_dimension = latent_dimension #size of the output of the trunk and branch network
        self.domain_dimension = domain_dimension # usually 2, since for the moment I am trying to solve the Poisson equation in 2d
        # the latent dimension is called p in the 2021 paper by Lu Jin and Karniadakis

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        mins = [x_min, y_min][:domain_dimension]
        maxs = [x_max, y_max][:domain_dimension] # generalization from just [min, max] which would still be valid in 2d
        self.coord_encoding_sincos = FourierEncoding(mins, maxs)

        # for the moment I am using two hidden layers. To be corrected later if necessary
        self.branch_network = nn.Sequential(
            nn.Linear(self.function_discretization_size, self.N_nodes_branch),  
            nn.Sigmoid(),        
            nn.Linear(self.N_nodes_branch, self.N_nodes_branch),  
            nn.Sigmoid(),   
            nn.Linear(self.N_nodes_branch, self.latent_dimension)
        )

        # after the Fourier encoding the trunk dimension doubles
        self.trunk_network = nn.Sequential(
            nn.Linear(2*self.domain_dimension, self.N_nodes_trunk),  
            nn.Tanh(),         
            nn.Linear(self.N_nodes_trunk, self.N_nodes_trunk),  
            nn.Tanh(),
            nn.Linear(self.N_nodes_trunk, self.N_nodes_trunk),  
            nn.Tanh(),   
            nn.Linear(self.N_nodes_trunk, self.latent_dimension)
        )

        # in Lu Jin Karniadakis they have branch depth 2, trunk depth 3 as standard

        # If I had not subclassed nn.Module, .parameters() would not be defined
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.to(self.device) 


    def forward(self, branch_input, trunk_input):
        """output of the model: scalar product between the outputs of the trunk and branch network"""

        branch_output = self.branch_network(branch_input)  
        trunk_input_2 = self.coord_encoding_sincos(trunk_input) 
        trunk_output  = self.trunk_network(trunk_input_2)      
        # outer product over the two batch dims, contracted on latent dimension

        return torch.einsum("bl,tl->bt", branch_output, trunk_output)


    def fit(self, num_epochs:int, branch_X, trunk_X, Y, branch_X_eval, trunk_X_eval, Y_eval, print_progress = True):
        # X, Y in the training dataset. 

        print("Y std across samples:", Y.std(dim=0).mean().item())
        print("Y std across points:", Y.std(dim=1).mean().item())
        
        self.epochs = []
        self.losses = []
        self.losses_eval = []
        self.rel_errors_eval = []

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

            with torch.no_grad(): # I do not need to track gradients here
                predictions_eval = self(branch_X_eval, trunk_X_eval)
                loss_eval = self.criterion(predictions_eval, Y_eval)
                rel_error_eval = torch.mean((predictions_eval - Y_eval)**2) / torch.mean(Y_eval**2)
            self.losses_eval.append(loss_eval.item())
            self.rel_errors_eval.append(rel_error_eval.item())

            if print_progress and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss eval: {loss_eval.item():.4f}")

                with torch.no_grad(): # check variations - is the model actually learning anything ?
                    preds = self(branch_X, trunk_X)
                    print("prediction std across samples:", preds.std(dim=0).mean().item())
                    print("prediction std across points:", preds.std(dim=1).mean().item())
                    


    def fit_from_npz(self, training_dataset:str, num_epochs:int, eval_rel_size = 0.2, print_progress = True):
    
            """Fits the model using a training dataset stored in a .npz file"""
            training_data = np.load(training_dataset)
    
            N_samples = training_data["f"].shape[0]
            N_train = int(N_samples * (1 - eval_rel_size))
    
            coords_branch = training_data["coords_branch"]
            coords_trunk = training_data["coords_trunk"]
            training_branch = training_data["f"][:N_train]
            training_Y = training_data["u"][:N_train]
    
            eval_branch = training_data["f"][N_train:]
            eval_Y = training_data["u"][N_train:]
    
            # this check is to ensure future reproducibility of the results
            if not np.array_equal(coords_branch, self.x_coords_for_branch):
                raise ValueError("Branch input mismatch! Function is not discretised on the same points as expected")
    
            # before training I need to convert the data to torch objects
            branch_X = torch.from_numpy(training_branch).float().to(self.device)
            trunk_X  = torch.from_numpy(coords_trunk).float().to(self.device)
            Y        = torch.from_numpy(training_Y).float().to(self.device)
    
            # to device, drectly here
            branch_X_eval = torch.from_numpy(eval_branch).float().to(self.device)
            Y_eval        = torch.from_numpy(eval_Y).float().to(self.device)
    
            self.fit(num_epochs, branch_X, trunk_X, Y, branch_X_eval = branch_X_eval, trunk_X_eval = trunk_X, Y_eval = Y_eval, print_progress=print_progress)

    
    def map_function_to_output_at_points(self, f: Callable, points_for_evaluation):

        branch = np.array([f(x[0], x[1]) for x in self.x_coords_for_branch])
        branch = torch.from_numpy(branch).float().unsqueeze(0).to(self.device)
        # unsqueeze(0) turns (N_points,) into (1, N_points)

        return self(branch, torch.from_numpy(points_for_evaluation).float().to(self.device))



# to do: mini-batching
# to do: implement ReduceLROnPlateau for eval/train samples
# to do: to improve convergence implement a scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
