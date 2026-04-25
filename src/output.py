import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_field(history, grid, interval=20, dt_plot=None, ylim=None, filename=None):
    """
    Animate the Ez field over time, with optional dielectric overlay and file saving.

    Parameters:
        history  : list of Ez snapshots, one per timestep
        grid     : Grid object
        interval : delay between frames in milliseconds
        ylim     : tuple (ymin, ymax), auto-scaled if None
        filename : if provided, save animation to this file (e.g. 'field.gif')
    """

    rc_fonts = {
        "font.family": "serif",
        "font.size": 12,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.serif": ["CMU Serif"]
    }
    mpl.rcParams.update(rc_fonts)

    history = np.array(history)
    x = np.arange(grid.size) * grid.dx

    fig, (ax_e, ax_m) = plt.subplots(2, 1, figsize=(10, 6), 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)

    # Top panel: Ez field
    line, = ax_e.plot(x, history[0])
    ax_e.set_ylabel('$E_{z}$ (V/m)')
    ax_e.set_title('FDTD 1D: $E_{z}$ field')
    if ylim is not None:
        ax_e.set_ylim(ylim)
    else:
        ax_e.set_ylim(history.min() * 1.2, history.max() * 1.2)
    step_text = ax_e.text(0.02, 0.95, '', transform=ax_e.transAxes)

    # Bottom panel: dielectric profile
    ax_m.plot(x, grid.eps, color='orange', label='$\epsilon_{\mathrm{r}}$')
    ax_m.plot(x, grid.mu,  color='green',  label='$\mu_{\mathrm{r}}$')
    ax_m.set_ylabel('$\epsilon_{\mathrm{r}}, \mu_{\mathrm{r}}$')
    ax_m.set_xlabel('$x$ (m)')
    ax_m.set_ylim(0, max(grid.eps.max(), grid.mu.max()) * 1.2)
    ax_m.legend(loc='upper right')

    plt.tight_layout()

    def update(frame):
        line.set_ydata(history[frame])
        step_text.set_text("")
        if (dt_plot is not None):
            step_text.set_text(f'$t = {frame * dt_plot * 1e9:.2f}$' + ' $\mathrm{ns}$')
        return line, step_text

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)
    plt.show()

    # Possible save the animation
    if filename is not None:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer=PillowWriter(fps=60))
        print("Animation saved.")

    return anim