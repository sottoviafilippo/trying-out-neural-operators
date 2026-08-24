# adapted by Claude starting from create_training_dataset_for_FNO.py

import numpy as np
import torch
from FFEM_building_blocks import Mesh
from pathlib import Path
from tqdm import tqdm


def generate_source_function(num_modes: int = 3, max_freq: int = 4, max_amplitude=1.):
    """
    Uses a random truncated Fourier series with random amplitudes
    """
    kx = np.random.randint(-(max_freq + 1), max_freq + 1, size=num_modes)
    ky = np.random.randint(-(max_freq + 1), max_freq + 1, size=num_modes)

    amplitudes = np.random.uniform(-max_amplitude, max_amplitude, size=num_modes)
    phases = np.random.uniform(0, 2 * np.pi, size=num_modes)

    def source_function(x, y):
        result = 0
        for i in range(num_modes):
            result += amplitudes[i] * np.cos(kx[i] * x + ky[i] * y + phases[i])
        return result

    return source_function


N_points_x = 48
N_points_y = 48

xx = np.linspace(-1, 1, N_points_x)
yy = np.linspace(-1, 1, N_points_y)
mymesh = Mesh(xx, yy, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()
X, Y = np.meshgrid(xx, yy, indexing="ij")  # each shape (N_points_x, N_points_y)

diri = lambda x, y: 0  # dirichlet boundary conditions

N_samples = 1000
N_train = 800  # rest goes to test

print(N_points_x, N_points_y)

f_all = np.zeros((N_samples, N_points_x, N_points_y))
u_all = np.zeros((N_samples, N_points_x, N_points_y))


for i in tqdm(range(N_samples)):
    func = generate_source_function()
    res_finite_elements = mymesh.run_simulation_poisson_dirichlet(func, diri)

    f_all[i] = func(xx, yy)
    u_all[i] = np.asarray(res_finite_elements)


X_b = np.broadcast_to(X, (N_samples, N_points_x, N_points_y))
Y_b = np.broadcast_to(Y, (N_samples, N_points_x, N_points_y))
# now stack X, Y, and the function values together, along the last dimension
input_all = np.stack([f_all, X_b, Y_b], axis=-3)  # (N, 3, N_points_x, N_points_y)
input_all = torch.from_numpy(input_all).float()

u_all = torch.from_numpy(u_all).float().unsqueeze(1)  # (N, 1, H, W)

input_train, input_test = input_all[:N_train], input_all[N_train:]
y_train, y_test = u_all[:N_train], u_all[N_train:]

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

torch.save({'x': input_train, 'y': y_train}, out_dir / "poisson_train_48.pt")
torch.save({'x': input_test, 'y': y_test}, out_dir / "poisson_test_48.pt")

print(f"Saved train: x {input_train.shape}, y {y_train.shape}")
print(f"Saved test:  x {input_test.shape}, y {y_test.shape}")

#print(input_all[3])
#print("x", X)
#print("y", Y)