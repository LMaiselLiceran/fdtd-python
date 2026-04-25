import numpy as np

class FDTDSolver1D:

	def __init__(self, grid, sources=None, boundaries=None):
		"""
		Initialize a 1D solver with appropriate sources and boundary conditions
		Also check the Courant condition
		"""
		self.grid = grid
		self.sources = sources or []
		self.boundaries = boundaries or []
		self._check_Courant()

	def _check_Courant(self):
		"""Check the Courant condition taking into account the slowest propagation speed"""
		g = self.grid
		c_min = g.c0 / np.sqrt(np.max(g.eps * g.mu))
		courant = c_min * g.dt / g.dx
		if (courant > 1.0):
			raise ValueError(
				f"Courant condition violated: C = {courant:.3f} > 1. "
				f"Reduce dt or increase dx."
			)

	def step(self, n):
		"""
		Perform a single time step. Update H, then E, then apply BCs and sources
		
		Parameters:
			n: number of the step (such that the physical time is n * dt)
		"""
		self._update_fields()
		for boundary in self.boundaries:
			boundary.apply(self.grid)
		for source in self.sources:
			source.apply(self.grid, self.grid.dt * n)

	def _update_fields(self):
		"""
		Update the magnetic and electric fields according to the discretized
		Faraday and Ampere laws, respectively.
		Takes into account the leapfrogging between the electric and magnetic fields
		"""
		g = self.grid
		g.Hy[:-1] -= (g.dt / (g.mu0  * g.mu[:-1] * g.dx)) * (g.Ez[1:] - g.Ez[:-1])
		g.Ez[1:]  -= (g.dt / (g.eps0 * g.eps[1:] * g.dx)) * (g.Hy[1:] - g.Hy[:-1])