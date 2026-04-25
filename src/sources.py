import numpy as np

def gaussian_pulse(t, t0, sigma, amplitude=1.0):
    """
    Returns the value of a Gaussian pulse at time t
    
    Parameters:
    - t: time at which the pulse is evaluated
    - t0: time at which the pulse is maximal
    - sigma: width of the pulse (in time)
    - amplitude (optional): height of the pulse at t0
    """
    return amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

class SoftSource:
    def __init__(self, grid, position, waveform_func):
        """
        Create a soft source (i.e., a source that adds to the field instead
        of setting its value)

        Parameters:
        - grid: the desired grid object
        - position: physical coordinate in metres
        - waveform_func: callable f(n) returning field value at timestep n
        """
        self.index = int(position / grid.dx)
        self.waveform = waveform_func

    def apply(self, grid, n):
        """Inject source into Ez at the source position"""
        grid.Ez[self.index] += self.waveform(n)