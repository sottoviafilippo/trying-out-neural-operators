import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# following the concepts presented in https://arxiv.org/abs/2512.01421v2 (*)
# also looking at https://arxiv.org/pdf/2010.08895 (**)


class FourierLayer(nn.Module):

    def __init__(self, hidden_dimension: int, n_modes: list):
        super().__init__()

        self.n_modes = n_modes
        #self.hidden_dimension = hidden_dimension
        # Nyquist-Shannon theorem: n_modes should not be > spatial res/2 
        
        # the weights for the Fourier part
        # n_modes appears twice because we are working on a 2d domain. For a 3d domain I would have , n_modes[0], n_modes[1], n_modes[2], ... etc.
        # same number of modes in both directions
        self.spectral_weight = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension, n_modes[0], n_modes[1], dtype=torch.cfloat) * 0.02)

        # local/skip path (linear) 
        self.channel_mixing = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, x):
        # Following (*), page 35. different structure compared to the paper, here channel last
        # FOR A REAL-VALUED SYSTEM rfft CAN BE USED TO SPEED UP THE CALCULATION: TO BE IMPLEMENTED

        # First do the FFT of the input tensor, in both directions
        fft_x = torch.fft.fftn(x, dim = (-3, -2))
        # implementation details: shuffle the order of the components so that the 0-mode is in the center
        fft_x = torch.fft.fftshift(fft_x, dim = (-3, -2))
        Nx = x.shape[-3]
        Ny = x.shape[-2]

        # Now select the lower n_modes modes, in both directions
        x_fft_selected = fft_x[:, Nx//2 - self.n_modes[0]//2 : Nx//2 + (self.n_modes[0] + 1)//2, Ny//2 - self.n_modes[1]//2 : Ny//2 + (self.n_modes[1] + 1)//2, :]

        # output initialization (frequency domain). Need to add device = x.device for out_fft not to go back to cpu even if I am using mps
        out_fft = torch.zeros(x.shape, dtype=torch.cfloat, device=x.device)
        # now ricombine over the channels (convolution across channels)
        out_fft[:, Nx//2 - self.n_modes[0]//2 : Nx//2 + (self.n_modes[0] + 1)//2, Ny//2 - self.n_modes[1]//2 : Ny//2 + (self.n_modes[1] + 1)//2, :] = torch.einsum('bxyi,ioxy->bxyo', x_fft_selected, self.spectral_weight)
        # now reshuffle to original order
        out_fft = torch.fft.ifftshift(out_fft, dim = (-3, -2))

        spectral_convolution_out = torch.fft.ifftn(out_fft, dim = (-3, -2))

        channel_mixing_out = self.channel_mixing(x)

        # return the real part (GELU is only implemented for floating types)
        return spectral_convolution_out.real + channel_mixing_out 



class FNO_v1(nn.Module):
    """Defines and sets up a simple FNO. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed
    # note: for the moment I am working on the [-1, 1] square. for general case better to normalize the coordinates

    def __init__(self, n_modes, hidden_dimension, input_dimension = 1, output_dimension = 1):
        # For the moment this only works for 2d problems

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        super().__init__() # refers to nn.Module

        self.n_modes = n_modes
        self.hidden_dimension = hidden_dimension
        self.input_dimension  = input_dimension
        self.output_dimension = output_dimension


        self.network = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimension), # lifting, no activation function needed after lifting - direct to Fourier layer
            FourierLayer(self.hidden_dimension, self.n_modes), # Fourier layer 1
            nn.GELU(), # according to (*) GELU 'has been shown to work well in smooth operator learning tasks'
            FourierLayer(self.hidden_dimension, self.n_modes), # Fourier layer 2
            nn.GELU(),
            nn.Linear(self.hidden_dimension, self.hidden_dimension), # projection layer 1
            nn.GELU(),
            nn.Linear(self.hidden_dimension, self.output_dimension) # projection layer 1
        )


        # Need to subclass nn.Module for .parameters() to be defined
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.to(self.device) 


    def forward(self, x):
        """output of the model"""
        
        return self.network(x)


    def fit(self, num_epochs:int, X, Y, X_eval, Y_eval, print_progress = True):
        # X, Y in the training dataset 

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

            if print_progress and (epoch + 1) % 10 == 0:
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

        # X: input, Y: output
        training_X = training_data["f"][:N_train]
        training_Y = training_data["u"][:N_train]

        eval_X = training_data["f"][N_train:]
        eval_Y = training_data["u"][N_train:]

        # before training I need to convert the data to torch objects
        # unsqueeze(1) makes the X and Y shape to (N_samples, N_x, N_y, 1), as needed (1 because I am solving for scalar functions sofar)
        X_train = torch.from_numpy(training_X).float().unsqueeze(-1).to(self.device)
        Y_train = torch.from_numpy(training_Y).float().unsqueeze(-1).to(self.device)

        # to device, directly here
        X_eval = torch.from_numpy(eval_X).float().unsqueeze(-1).to(self.device)
        Y_eval = torch.from_numpy(eval_Y).float().unsqueeze(-1).to(self.device)

        self.fit(num_epochs, X_train, Y_train, X_eval, Y_eval, print_progress=print_progress)


    def map_function_to_output_at_points(self, f: Callable, points_for_evaluation):

        # TO DO

        return 0
