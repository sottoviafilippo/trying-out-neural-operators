import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np
from neuralop.training import AdamW

# following the concepts presented in https://arxiv.org/abs/2512.01421v2 (*)
# also looking at https://arxiv.org/pdf/2010.08895 (**) (original FNO paper)


class FourierLayer(nn.Module):

    # see (*): in the literature the FourierLayer is more complex, with one additional skip connection

    def __init__(self, hidden_dimension: int, n_modes: list):
        super().__init__()

        self.n_modes = n_modes
        #self.hidden_dimension = hidden_dimension
        # Nyquist-Shannon theorem: n_modes should not be > spatial res/2 
        
        # the weights for the Fourier part
        # n_modes appears twice because we are working on a 2d domain. For a 3d domain I would have , n_modes[0], n_modes[1], n_modes[2], ... etc.
        # same number of modes in both directions
        self.spectral_weight = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension, n_modes[0], n_modes[1], dtype=torch.cfloat) / hidden_dimension**2)

        # local/skip path (linear) 
        self.channel_mixing = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, x):
        # Following (*), page 35. different structure compared to the paper, here channel last

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

        channel_mixing_out = self.channel_mixing(x) # skip connection

        # return the real part (GELU is only implemented for floating types)
        return spectral_convolution_out.real + channel_mixing_out 


class FourierLayer_real(nn.Module):
    # For real data, using rfftn for improved. Not using fftshift, also for better speed

    def __init__(self, hidden_dimension: int, n_modes: list):
        super().__init__()
        self.n_modes = n_modes
        # Nyquist-Shannon: n_modes[i] should not exceed spatial_res // 2

        # rfftn only omits neg freqs along the last transformed dim, which here corresponds to the y-direction
        # https://docs.pytorch.org/docs/2.13/generated/torch.fft.rfftn.html
        # so one needs to separate weight blocks: one small positive kx and one small negative kx (wraparound), both with small ky
        self.spectral_weight_pos = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension, n_modes[0], n_modes[1], dtype=torch.cfloat) / hidden_dimension**2) 
        # the scale factor would be 1/(in_channel * out_channel) , here it reduces to (1./hidden_dimension)**2
        self.spectral_weight_neg = nn.Parameter(torch.randn(hidden_dimension, hidden_dimension, n_modes[0], n_modes[1], dtype=torch.cfloat) / hidden_dimension**2)

        self.channel_mixing = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, x):
        Nx = x.shape[-3]
        Ny = x.shape[-2]
        m0, m1 = self.n_modes

        # real FFT: last transformed dim (-2) collapses to Ny//2 + 1 non-negative frequencies
        fft_x = torch.fft.rfftn(x, dim=(-3, -2))

        out_fft = torch.zeros(x.shape[0], Nx, Ny // 2 + 1, x.shape[-1],dtype=torch.cfloat, device=x.device)

        # small positive kx, small ky
        out_fft[:, :m0, :m1, :] = torch.einsum('bxyi,ioxy->bxyo', fft_x[:, :m0, :m1, :], self.spectral_weight_pos)
        # small negative kx (wraps to the tail of the array), small ky
        out_fft[:, -m0:, :m1, :] = torch.einsum('bxyi,ioxy->bxyo', fft_x[:, -m0:, :m1, :], self.spectral_weight_neg)

        spectral_convolution_out = torch.fft.irfftn(out_fft, s=(Nx, Ny), dim=(-3, -2)) # Kl in (*) page 32
        channel_mixing_out = self.channel_mixing(x) # Wl in (*) page 32

        return spectral_convolution_out + channel_mixing_out # don't see spectral_convolution_out.real because it's already real 


class FNO_v1(nn.Module):
    """Defines and sets up a simple FNO. first approach: trained on solutions of the 2d Poisson equation"""
    # (for the moment) the sampling points of the input functions are fixed
    # note: for the moment I am working on the [-1, 1] square. for general case better to normalize the coordinates

    def __init__(self, n_modes, hidden_dimension, input_dimension = 3, output_dimension = 1, lr = 0.004):
        # For the moment this only works with 2d problems
        # default input dimension is 3: x, y, function_value
        # default output dimension is 1 since we are outputting the values of a scalar function at given cooordinates

        super().__init__() # refers to nn.Module

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # use gpu if possible (mac)

        self.n_modes = n_modes
        self.hidden_dimension = hidden_dimension # the authors of (*) recommend starting with a number of hidden channels 16-32
        self.input_dimension  = input_dimension
        self.output_dimension = output_dimension

        self.network = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimension), # lifting, no activation function needed after lifting - direct to Fourier layer
            FourierLayer_real(self.hidden_dimension, self.n_modes), # Fourier layer 1
            nn.GELU(), # according to (*) GELU 'has been shown to work well in smooth operator learning tasks'
            FourierLayer_real(self.hidden_dimension, self.n_modes), # Fourier layer 2
            nn.GELU(),
            FourierLayer_real(self.hidden_dimension, self.n_modes), # Fourier layer 3. According to (*) advisable to start with 3-6 layers and then eventually increase
            nn.GELU(),
            nn.Linear(self.hidden_dimension, self.hidden_dimension), # projection layer 1
            nn.GELU(),
            nn.Linear(self.hidden_dimension, self.output_dimension) # projection layer 1
        )

        # Need to subclass nn.Module for .parameters() to be defined
        
        #self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay = 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        """self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30) # CosineAnnealing: smooth decay of the learning rate"""

        self.criterion = nn.MSELoss()

        self.to(self.device) 


    def forward(self, x):
        """output of the model"""
        
        return self.network(x)


    def fit(self, num_epochs:int, X, Y, X_eval, Y_eval, batch_size = 32, print_progress = True):
        # X, Y in the training dataset 

        print("Y std across samples:", Y.std(dim=0).mean().item())
        print("Y std across points: ", Y.std(dim=1).mean().item())
        
        self.epochs = []
        self.losses = []
        self.losses_eval = []
        self.rel_errors_eval = []

        N = X.shape[0] # number of samples in training dataset

        for epoch in range(num_epochs):

            # first shuffle indices to prepare the minibatches
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0

            for i in range(0, N, batch_size): # cut up the training data in slices of size batch_size 
                idx = perm[i:i+batch_size]
                X_batch = X[idx]
                Y_batch = Y[idx]

                predictions_batch = self(X_batch)
                loss_batch = self.criterion(predictions_batch, Y_batch)

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step() # update the weights

                epoch_loss += loss_batch.item() * X_batch.shape[0]
    
            self.scheduler.step()
            epoch_loss = epoch_loss / N

            # Track the loss history
            self.epochs.append(epoch + 1)
            self.losses.append(epoch_loss)

            with torch.no_grad(): # I do not need to track gradients here
                predictions_eval = self(X_eval)
                loss_eval = self.criterion(predictions_eval, Y_eval)
                rel_error_eval = torch.mean((predictions_eval - Y_eval)**2) / torch.mean(Y_eval**2)
            self.losses_eval.append(loss_eval.item())
            self.rel_errors_eval.append(rel_error_eval.item())

            if print_progress and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Loss eval: {loss_eval.item():.4f}")

                with torch.no_grad(): # check variations - is the model actually learning anything ?
                    preds = self(X)
                    print("prediction std across samples:", preds.std(dim=0).mean().item())
                    print("prediction std across points:", preds.std(dim=1).mean().item())
                    


    def fit_from_npz(self, training_dataset:str, num_epochs:int, eval_rel_size = 0.2, print_progress = True):

        """Fits the model using a training dataset stored in a .npz file"""
        training_data = np.load(training_dataset)

        N_samples = training_data["input"].shape[0]
        N_train = int(N_samples * (1 - eval_rel_size))

        # X: input, Y: output
        training_X = training_data["input"][:N_train]
        training_Y = training_data["u"][:N_train]

        eval_X = training_data["input"][N_train:]
        eval_Y = training_data["u"][N_train:]

        # before training I need to convert the data to torch objects
        # unsqueeze(1) makes the Y shape to (N_samples, N_x, N_y, 1), as needed (1 because I am solving for scalar functions sofar)
        # with 3 dims (x, y, func_value), on the other hand, X already has the right dimensions
        X_train = torch.from_numpy(training_X).float().to(self.device)
        Y_train = torch.from_numpy(training_Y).float().unsqueeze(-1).to(self.device)

        # to device, directly here
        X_eval = torch.from_numpy(eval_X).float().to(self.device)
        Y_eval = torch.from_numpy(eval_Y).float().unsqueeze(-1).to(self.device)

        self.fit(num_epochs, X_train, Y_train, X_eval, Y_eval, print_progress=print_progress)


    def map_function_to_output_at_points(self, f: Callable, X: np.ndarray, Y: np.ndarray):
        # X, Y: from np.meshgrid
        F = f(X, Y)

        grid_input = np.stack([X, Y, F], axis=-1)  # (Nx, Ny, 3)
        grid_input = torch.from_numpy(grid_input).float().unsqueeze(0).to(self.device)  # dimensions: (1, Nx, Ny, 3)

        with torch.no_grad():
            grid_output = self(grid_input).squeeze(0).squeeze(-1).cpu().numpy()  

        return grid_output
