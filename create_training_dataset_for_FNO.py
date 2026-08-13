import numpy as np
from FFEM_building_blocks import Mesh
from pathlib import Path


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
    


xx = np.linspace(-1, 1, 64)
yy = np.linspace(-1, 1, 64)
mymesh = Mesh(xx, yy, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()
# coordinates of mesh points 

coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
# in this way I create a list will all sets of 2d cordinates
diri = lambda x, y: 0 # (0) dirichlet boundary conditions to be used for the moment

x_branch = np.linspace(-1, 1, 20)
y_branch = np.linspace(-1, 1, 20)
# dont need branch to be too large, a couple hundred points should work (to be checked!)
X_branch, Y_branch = np.meshgrid(x_branch, y_branch, indexing="ij")
coords_branch = np.stack([X_branch.ravel(), Y_branch.ravel()], axis=-1)

"""
strategy: prioritize sample diversity to spatial distribution
to use (simple for the moment): random polynomials, random Fourier series
"""


N_source_functions = 1000 # number of source functions for which we compute the solution
N_points = coords.shape[0]

#vectors to store the functions (evaluated at the grid) and the corresponding solutions
f_all = np.zeros((N_source_functions, N_points)) # discretization of the functions
u_all = np.zeros((N_source_functions, N_points)) # will contain the desired output of the model

for i in range(N_source_functions):
    func = generate_source_function()
    res_finite_elements = mymesh.run_simulation_poisson_dirichlet(func, diri)

    f_all[i] = func(xx, yy).ravel()
    u_all[i] = np.asarray(res_finite_elements).ravel()

    if i%50 == 0:
        print(f"[{i+1}/{N_source_functions}] done")

out_path = Path("data/dataset_for_fno.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

# save the dataset in the .npz file
np.savez(out_path,
    coords = coords, # (N_points, 2)
    f = f_all,          # (N_samples, N_points)
    u = u_all,          # (N_samples, N_points)
)

print(f"Saved at {out_path}, for coords_branch={coords_branch.shape}, f has shape {f_all.shape}, u has shape n{u_all.shape}")
