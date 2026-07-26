import numpy as np
from FFEM_building_blocks import Mesh
from pathlib import Path

# to understnad : need to normalise or not ?? up to how large of a source function can I get??
# HOW MANY MODES ? WHT MAX FREQ?
def generate_source_function(num_modes:int=4, max_freq:int=5, max_amplitude = 1.):
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
    


x_trunk = np.linspace(-1, 1, 40)
y_trunk = np.linspace(-1, 1, 40)
mymesh = Mesh(x_trunk, y_trunk, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()
# coordinates of mesh points 
X_trunk, Y_trunk = np.meshgrid(x_trunk, y_trunk, indexing="ij")
coords_trunk = np.stack([X_trunk.ravel(), Y_trunk.ravel()], axis=-1)
# in this way I create a list will all sets of 2d cordinates
diri = lambda x, y: 0 # (0) dirichlet boundary conditions to be used for the moment

x_branch = np.linspace(-1, 1, 20)
y_branch = np.linspace(-1, 1, 20)
# dont need branch to be too large, a couple hundred points should word (to be checked!)
X_branch, Y_branch = np.meshgrid(x_branch, y_branch, indexing="ij")
coords_branch = np.stack([X_branch.ravel(), Y_branch.ravel()], axis=-1)

"""
strategy: prioritize sample diversity to spatial distribution
to use (simple for the moment): random polynomials, random Fourier series
"""


N_source_functions = 7000 # number of source functions for which we compute the solution
N_points_trunk  = coords_trunk.shape[0]
N_points_branch = coords_branch.shape[0]

#vectors to store the functions (evaluated at the grid) and the corresponding solutions
f_all = np.zeros((N_source_functions, N_points_branch)) # discretization of the functions
u_all = np.zeros((N_source_functions, N_points_trunk)) # will contain the desired output of the model

for i in range(N_source_functions):
    func = generate_source_function()
    res_finite_elements = mymesh.run_simulation_poisson_dirichlet(func, diri)

    f_all[i] = func(X_branch, Y_branch).ravel()
    u_all[i] = np.asarray(res_finite_elements).ravel()

    if i%50 == 0:
        print(f"[{i+1}/{N_source_functions}] done")

out_path = Path("data/dataset.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

# save the dataset in the .npz file
np.savez(out_path,
    coords_branch = coords_branch, # (N_points_branch, 2)
    coords_trunk  = coords_trunk,   # (N_points_trunk, 2)
    f = f_all,          # (N_samples, N_points_branch)
    u = u_all,          # (N_samples, N_points_trunk)
)

print(f"Saved at {out_path}, for coords_branch={coords_branch.shape}, f has shape {f_all.shape}, u has shape n{u_all.shape}")
