import warnings

def add_material_slab(grid, x_start, x_end, eps_r=1.0, mu_r=1.0):
    """Add a dielectric slab to the grid between x_start and x_end (in metres)"""

    # If starting and ending positions are flipped, warn the user and flip them back
    if (x_start > x_end):
        warnings.warn(f"x_start ({x_start}) is larger than x_end ({x_end}). Flipping them.")
        x_start, x_end = x_end, x_start

    # If slab is out of bounds, clip and warn the user
    if (x_start < 0) or (x_end > grid.L):
        warnings.warn(f"Slab [{x_start}, {x_end}] is out of domain [0, {grid.L}]. Clipping to domain.")
        x_start = max(x_start, 0)
        x_end   = min(x_end, grid.L)

    # If slab is too short, warn the user
    if (x_end - x_start < grid.dx):
        warnings.warn(f"Slab [{x_start}, {x_end}] is smaller than dx. Behavior might be unexpected.")

    # Allocate material properties
    i_start = round(x_start / grid.dx)
    i_end   = round(x_end   / grid.dx)
    grid.eps[i_start:i_end] = eps_r
    grid.mu[i_start:i_end]  = mu_r