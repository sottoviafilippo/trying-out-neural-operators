import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# following the concepts presented in https://arxiv.org/abs/2512.01421v2


class FourierLayer(nn.Module):

    def __init__(self, n_modes):
        super().__init__()

        self.n_modes = n_modes
        # Nyquist-Shannon theorem: n_modes should not be > spatial res/2 
        
    def forward(self, x):

        # TO DO 
        return 0


class LiftingLayer(nn.Module):

    def __init__(self, hidden_dimension):
        super().__init__()
        self.hidden_dimension = hidden_dimension

        
    def forward(self, x):

        # TO DO 
        return 0


class ProjectionLayer(nn.Module):

    def __init__(self, n_modes):
        super().__init__()

        
    def forward(self, x):

        # TO DO 
        return 0



class FNO_v1(nn.Module):
    """Defines and sets up a simple FNO. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed
    # note: for the moment I am working on the [-1, 1] square. for general case better to normalize the coordinates

    def __init__(self,  n_modes, hidden_dimension, domain_dimension:int = 2):

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        super().__init__() # refers to nn.Module

        self.n_modes = n_modes
        self.hidden_dimension = hidden_dimension


        self.network = nn.Sequential(
            LiftingLayer(self.hidden_dimension),
            FourierLayer(self.n_modes),
            ProjectionLayer
        )


        # If I had not subclassed nn.Module .parameters() would not be defined
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.to(self.device) 


    def forward(self, x):
        """output of the model"""
        
        return self.network(x)


    def fit(self, num_epochs:int, X, Y, X_eval, Y_eval, print_progress = True):
        # X, Y in the training dataset. 

        print("Y std across samples:", Y.std(dim=0).mean().item())
        print("Y std across points: ", Y.std(dim=1).mean().item())
        
        self.epochs = []
        self.losses = []
        self.losses_eval = []
        self.rel_errors_eval = []


        for epoch in range(num_epochs):
    
            predictions = self(X)  # __call__ "does some bookkeeping" and calls the previously defined self.forward()
            loss = self.criterion(predictions, Y)

            self.optimizer.zero_grad() # otherwise the gradients would be added to the previously computed ones
            loss.backward() # computes the gradient of the loss compute on the neural network, via model(X)
            self.optimizer.step() # updates the weights

            # Track the loss history
            self.epochs.append(epoch + 1)
            self.losses.append(loss.item())

            with torch.no_grad(): # I do not need to track gradients here
                predictions_eval = self(X_eval)
                loss_eval = self.criterion(predictions_eval, Y_eval)
                rel_error_eval = torch.mean((predictions_eval - Y_eval)**2) / torch.mean(Y_eval**2)
            self.losses_eval.append(loss_eval.item())
            self.rel_errors_eval.append(rel_error_eval.item())

            if print_progress and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss eval: {loss_eval.item():.4f}")

                with torch.no_grad(): # check variations - is the model actually learning anything ?
                    preds = self(X)
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

        # TO DO

        return 0
