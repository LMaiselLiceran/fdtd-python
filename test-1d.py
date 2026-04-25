from src.grid import Grid
from src.solver import FDTDSolver1D
from src.sources import SoftSource, gaussian_pulse
from src.boundaries import SimpleABC
from src.materials import add_material_slab
from src.output import animate_field
from functools import partial
import numpy as np

# Set up
grid = Grid(L=0.2, dx=1e-3, courant=0.5)
add_material_slab(grid, x_start=0.02, x_end=0.05, eps_r=1.0, mu_r=4.0)
add_material_slab(grid, x_start=0.13, x_end=0.18, eps_r=8.0, mu_r=1.0)

# Define the waveforms with fixed parameters, leaving only n free
wf1  = partial(gaussian_pulse, t0=5e-10, sigma=1e-10, amplitude=400)
src1 = SoftSource(grid, position=0.1, waveform_func=wf1)

wf2  = partial(gaussian_pulse, t0=15e-10, sigma=0.5e-10, amplitude=200)
src2 = SoftSource(grid, position=0.15, waveform_func=wf2)

wf3  = partial(gaussian_pulse, t0=20e-10, sigma=2e-10, amplitude=400)
src3 = SoftSource(grid, position=0.04, waveform_func=wf3)

abc = SimpleABC()
abc.setup(grid)
solver = FDTDSolver1D(grid, [src1, src2, src3], [abc])

# Time setup
t_end   = 5e-9		# run for a total of 10 nanoseconds
dt_plot = 5e-12		# save a frame every 5 picoseconds
n_steps = round(t_end / grid.dt)

# Run
history = []
history_plot = []
next_plot_time = 0.0

for n in range(n_steps):

    solver.step(n)
    t = n * grid.dt
    history.append(grid.Ez.copy())

    if (t >= next_plot_time):
        history_plot.append(grid.Ez.copy())
        next_plot_time += dt_plot

# Visualize
# anim = animate_field(history, grid, ylim=(-1500, 1500), filename="field.gif")
anim = animate_field(history_plot, grid, ylim=(-2000, 2000), interval=1, dt_plot=dt_plot)