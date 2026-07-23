import numpy as np
from FFEM_building_blocks import Mesh
from pathlib import Path


def generate_source_function():
    return lambda x, y: 0


x = np.linspace(0, 2, 75)
y = np.linspace(0, 2, 75)
mymesh = Mesh(x, y, verbose=True)
mymesh.build_mass_matrix()
mymesh.build_stiffness_matrix()
# coordinates of mesh points 
X, Y = np.meshgrid(x, y, indexing="ij")
coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
diri = lambda x, y: 0


"""
strategy: prioritize sample diversity to spatial distribution
to use (simple for the moment): random polynomials, random Fourier series
"""


N_source_functions = 500 # number of source functions for which we compute the solution
N_points = coords.shape[0]

#vectors to store the functions (evaluated at the grid) and the corresponding solutions
f_all = np.zeros((N_source_functions, N_points))
u_all = np.zeros((N_source_functions, N_points))

for i in range(N_source_functions):
    func = generate_source_function()
    res_finite_elements = mymesh.run_simulation_poisson_dirichlet(func, diri)

    f_all[i] = func(X, Y).ravel()
    u_all[i] = np.asarray(res_finite_elements).ravel()

    if i%50 == 0:
        print(f"[{i+1}/{N_source_functions}] done")

out_path = Path("data/dataset.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(out_path,
    coords=coords,   # (N_points, 2)
    f=f_all,          # (N_samples, N_points)
    u=u_all,          # (N_samples, N_points)
)

print(f"Saved at {out_path}, for coords={coords.shape}, f has shape {f_all.shape}, u has shape n{u_all.shape}")
