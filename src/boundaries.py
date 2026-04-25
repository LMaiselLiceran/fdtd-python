import numpy as np

class SimpleABC:

	def __init__(self):
		"""Initialize a set of absorbing boundary conditions"""

		# Store field values at the two cells adjacent to each boundary
		self.left_old  = [0.0, 0.0]		# E[0] and E[1] from previous step
		self.right_old = [0.0, 0.0]		# E[-2] and E[-1] from previous step

		# Initialize coefficients for Mur's ABC formula
		self.Mur_coeff_left  = None
		self.Mur_coeff_right = None

	def setup(self, grid):
		"""Compute the Mur coefficients at both sides of the domain"""

		c_left  = grid.c0 / np.sqrt(grid.eps[0]  * grid.mu[0])
		c_right = grid.c0 / np.sqrt(grid.eps[-1] * grid.mu[-1])

		self.Mur_coeff_left  = (c_left  * grid.dt - grid.dx) / (c_left  * grid.dt + grid.dx)
		self.Mur_coeff_right = (c_right * grid.dt - grid.dx) / (c_right * grid.dt + grid.dx)

	def apply(self, grid):
		"""Update Ez at the endpoints according to Mur's ABC formula"""

		# Make sure the boundary conditions were properly set up
		if self.Mur_coeff_left is None:
			raise RuntimeError("SimpleABC.setup() must be called before apply()")

		E = grid.Ez

		# Left boundary
		E[0] = self.left_old[1] + self.Mur_coeff_left * (E[1] - self.left_old[0])

		# Right boundary
		E[-1] = self.right_old[0] + self.Mur_coeff_right * (E[-2] - self.right_old[1])

		# Save current values for next time step
		self.left_old  = [E[0], E[1]]
		self.right_old = [E[-2], E[-1]]