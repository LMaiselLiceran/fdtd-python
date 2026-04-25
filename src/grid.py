import numpy as np
import warnings

class Grid:

    def __init__(self, L, dx, dt=None, courant=None):
        """
        Creates a grid object for the simulation

        Parameters:
            L: length of the domain (in meters)
            dx: spatial step (in meters)
            dt (optional): time step (in seconds)
            courant (optional): courant parameter c0 * dt / dx < 1
        Note: exactly one of either dt or courant must be provided
        """

        # Make sure at least either dt or courant was provided
        if (dt is None) and (courant is None):
            raise ValueError("Provide exactly one of dt or courant, not both or neither.")

        # If both were provided, warn the user that dt will not be used
        if (dt is not None) and (courant is not None):
            warnings.warn("Warning: both dt and courant provided, disregarding dt")
        
        # Define physical constants (in SI units)
        self.c0 = 299792458.0
        self.eps0 = 8.854187817e-12
        self.mu0  = 4 * np.pi * 1e-7

        # Define coordinate step, number of points, and adjust domain length
        self.dx = dx
        self.size = round(L / dx)
        self.L = self.size * dx  # recompute to ensure consistency
        if abs(self.L - L) > 0.01 * dx:
            warnings.warn(f"L = {L} m is not a multiple of dx = {dx} m. "
                          f"Adjusted to L = {self.L} m.")

        # Define courant and dt, calculated depending on what was provided
        if courant is not None:
            self.courant = courant
            self.dt = courant * dx / self.c0
        else:
            self.dt = dt
            self.courant = self.c0 * self.dt / self.dx

        # Initialize fields and vacuum background
        self.Ez = np.zeros(self.size)    # electric field
        self.Hy = np.zeros(self.size)    # magnetic field
        self.eps = np.ones(self.size)    # relative permittivity
        self.mu = np.ones(self.size)     # relative permeability