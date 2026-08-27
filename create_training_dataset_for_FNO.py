import numpy as np
from FFEM_building_blocks import Mesh
from pathlib import Path
from tqdm import tqdm


def generate_source_function(num_modes:int=3, max_freq:int=4, max_amplitude = 1.):
    """
    Uses a random truncated Fourier series with random amplitudes
    """
    # first pick some frequencies
    kx = np.random.randint(-(max_freq + 1), max_freq + 1, size=num_modes)
    ky = np.random.randint(-(max_freq + 1), max_freq + 1, size=num_modes)

    # now draw the corresponding amplitudes and phases
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
# coordinates of mesh points 

# 2D grid of coordinates, matching the (N_points_x, N_points_y) layout of f_all/u_all
X, Y = np.meshgrid(xx, yy, indexing="ij")  # each shape (N_points_x, N_points_y)

diri = lambda x, y: 0 # (0) dirichlet boundary conditions to be used for the moment


"""
strategy: prioritize sample diversity to spatial distribution
to use (simple for the moment): random polynomials, random Fourier series
"""

N_source_functions = 2000 # number of source functions for which we compute the solution

print(N_points_x, N_points_y)

#vectors to store the functions (evaluated at the grid) and the corresponding solutions
f_all = np.zeros((N_source_functions, N_points_x, N_points_y)) # discretization of the functions
u_all = np.zeros((N_source_functions, N_points_x, N_points_y)) # will contain the desired output of the model

for i in tqdm(range(N_source_functions)):
    func = generate_source_function()
    res_finite_elements = mymesh.run_simulation_poisson_dirichlet(func, diri)

    f_all[i] = func(xx, yy)
    u_all[i] = np.asarray(res_finite_elements)


# build the 3-channel FNO input: (x, y, f(x,y)) at every grid point
# numpy.broadcast_to(array, shape, subok=False): broadcast an array to a new shape https://numpy.org/devdocs/reference/generated/numpy.broadcast_to.html
X_b = np.broadcast_to(X, (N_source_functions, N_points_x, N_points_y))
Y_b = np.broadcast_to(Y, (N_source_functions, N_points_x, N_points_y))
# now stack X, Y, and the function values together, along the last dimension
input_all = np.stack([X_b, Y_b, f_all], axis=-1)  # (N_samples, N_points_x, N_points_y, 3)

out_path = Path("data/dataset_for_fno.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(out_path,
    input = input_all,  # (N_samples, N_points_x, N_points_y, 3) -> channels: x, y, f
    u = u_all,          # (N_samples, N_points_x, N_points_y)
)

print(f"Saved at {out_path}, f has shape {f_all.shape}, u has shape n{u_all.shape}")