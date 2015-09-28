from __future__ import division

import random
import csv

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import stats
from scipy.spatial import ConvexHull


#  Precalculate the valid moves for each compass direction NESW
moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
moves = [np.array(move) for move in moves]

def weighted_choice(weights):
	'''
	Given a list of numerical weights, return an index with weights[i]/sum(weights).
	Adapted from http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
	'''
	if all([w==0 for w in weights]):
		return random.randint(0, len(weights)-1)

	rnd = random.random() * sum(weights)
	for i, w in enumerate(weights):
		rnd -= w
		if rnd < 0:
			return i

def normalise_array(values):
	'''
	Given an input array of numericals, returns the
	array scaled between 1 and 0
	'''
	flattened_values = values.flatten()
	values -= min(flattened_values)
	values /= max(flattened_values)
	return values


class World(object):
	''' 
	A World contains an environment, and some animals.
	'''
	def __init__(self, environment_size, n_animals):
		self.environment = Environment(environment_size)
		self.animals = [Animal(self.environment) for i in xrange(n_animals)]

	def update(self, kind, n_iter=1, m=1, m1=1, m2=1):
		'''
		Advances the world.
		First updates the positions of the animals,
		then updates the quality of the environment.
			kind: type of environment quality
		'''

		for i in xrange(n_iter):
			for animal in self.animals:
				animal.move(self.environment, m)
			if kind in ('scent', 'combined'):
				self.environment.add_new_animal_scent(self.animals)
				self.environment.update_scent()
				self.environment.quality_global = self.environment.quality_basic**m1 + self.environment.quality_scent**m2


	def static_plot(self, convergence=False, quality_scent=False, quality_global=False, quality_basic=False, kde=False, history=False, history_scatter=False, location=False, convex_hull=False, save=False):
		'''
		Plot a single figure.
		Won't return until plot is closed.
		'''

		# Create figure objects for passing to plot functions.
		fig = plt.figure()
		ax = plt.axes(xlim=(0, self.environment.x_max), ylim=(self.environment.y_max))

		# Environment plots
		if quality_scent: 
			self.environment.plot_quality_scent(fig)
		if quality_basic: 
			self.environment.plot_quality_basic(fig)
		if quality_global:
			self.environment.plot_quality_global(fig)

		# Animal plots
		for animal in self.animals:
			if history: 
				animal.plot_history(fig)
			if history_scatter: 
				animal.plot_history_scatter(fig)


		# Overall plots
		if kde:
			self.plot_kde(fig) 
		if convex_hull:
			self.plot_convex_hull(fig)

		# Exoirt plot to a given file name
		if save:
			# plt.axes(frameon=False)
			ax.axes.get_yaxis().set_visible(False)
			ax.axes.get_xaxis().set_visible(False)
			fig.savefig(save, transparent=True, bbox_inches='tight')

		# Sneakily plot a second plot of convergence of the method, for assessing quality of the simulation
		if convergence:
			fig2 = plt.figure()
			self.plot_convergence(fig2)

		# Finnaly, display the plots
		plt.show()

	def plot_convergence(self, fig):
		'''
		PLot the convergence of the separation of the animals
		as time goes on, for assessing the quiality of the simulation
		'''

		# Need multiple animals for convergence
		if len(self.animals) < 2:
			return

		n_animals = len(self.animals)
		n_iter = len(self.animals[0].position_history)
		X = range(n_iter)
		Y = np.zeros(n_iter)

		for i in X:
			animal_positions = np.array([a.position_history[i] for a in self.animals])
			centroid = np.sum(animal_positions, 0) / n_animals
			mean_distance =  np.mean([np.linalg.norm(centroid - p) for p in animal_positions])
			Y[i] = mean_distance

		plt.plot(X, Y)

	def plot_convex_hull(self, fig):
		'''
		Plot minimum convex polygon of animal locations
		'''
		points = np.vstack([a.position_history for a in self.animals])
		hull = ConvexHull(points)
		for simplex in hull.simplices:
			plt.plot(points[simplex,0], points[simplex,1], 'k--', lw=2)

	def plot_kde(self, fig):
		'''
		Plot Kernel Density Estimaion of animal locations, with a 5 percent cutoff.
		'''


		hist = np.vstack([np.array([0, 0]), self.environment.size-1])

		for animal in self.animals:
			hist = np.vstack([hist, animal.position_history])

		hist = hist[::20]

		m1 = hist[:, 0]
		m2 = hist[:, 1]
		xmin = m1.min()
		xmax = m1.max()
		ymin = m2.min()
		ymax = m2.max()

		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		values = np.vstack([m1, m2])
		kernel = stats.gaussian_kde(values)
		P = kernel(positions)
		Z = np.reshape(P.T, X.shape)
		# plt.imshow(np.rot90(Z), cmap=plt.cm.Reds,
		# 		  extent=[xmin, xmax, ymin, ymax])

		cutoff = (max(P) - min(P)) * 0.05 + min(P)
		levels = [cutoff]

		plt.contour(X, Y, Z, levels=levels, linestyles="dashed", colors="k", linewidths=3, zorder=1000)


class Environment(object):
	'''
	A 2D area explored by animal, containing 2D qualities
	'''

	def __init__(self, size):
		'''
		Sets a size, but you still need to populate qualities before use
		'''
		
		if type(size) == int:
			size = (size, size) #square environment

		# Set a whole bunch of handy parameters, all derived from the size input
		self.size = np.array(size, dtype=int)
		self.size_x = self.size[0]
		self.size_y = self.size[1]
		self.x_max = self.size_x - 1
		self.y_max = self.size_y - 1		
		self.centre = (self.size-1)/2 
		self.centre_x = self.centre[0]
		self.centre_y = self.centre[1]
		self.origin = np.array([0, 0])
		self.quality_scent = np.zeros(shape=self.size)



	def add_no_quality(self):
		'''
		All locations have the same (zero) quality, for initialisation
		'''
		self.quality_basic = np.zeros(shape=self.size)
		self.quality_global = self.quality_basic

	def add_uniform_quality(self):
		'''
		All locations have the same quality, for random walk
		'''
		self.quality_basic = 0.5 * np.ones(shape=self.size)
		self.quality_global = self.quality_basic

	def add_gaussian_quality(self):
		'''
		Random gausian landscape (with baked in Gaussian coefficient)
		'''
		gaussian_coefficient = 0.1 * np.average(self.size)
		noise = np.random.random(self.size)
		landscape = ndimage.gaussian_filter(noise, gaussian_coefficient)
		landscape = normalise_array(landscape)
		self.quality_basic = landscape
		self.quality_global = self.quality_basic

	def add_decreasing_centred_quality(self):
		'''
		greatest weight in the centre, scaled to zero at the corners
		I=I0 - alhpa x
		'''
		self.quality_basic = np.zeros(shape=self.size)
		weight_max = 1
		max_d_from_centre = np.linalg.norm(self.origin - self.centre)
		alpha = weight_max / max_d_from_centre

		# iterate over all coordinates
		for x, y in np.ndindex(self.quality_basic.shape):
			position = np.array([x, y])
			d_from_centre = np.linalg.norm(position - self.centre)
			self.quality_basic[x, y] = weight_max - alpha*d_from_centre
		self.quality_global = self.quality_basic

	def add_new_animal_scent(self, animals):
		'''
		Add new scent marking after an animal has moved
		'''
		scent_amount = 1
		for animal in animals:
			previous_position = animal.position_history[-2, :]
			self.quality_scent[previous_position[0], previous_position[1]] = scent_amount
			self.quality_global = self.quality_basic + self.quality_scent

	def update_scent(self):
		'''
		Updates the quality_scent of the environment, based
		on diffusion of the current sent by the explicit finite volume method.
		'''
		dx = 0.03
		dt = 0.1
		SP = 0.1
		D = 0.0005
		aP0 = dx / dt
		aP = aP0
		aN, aS, aW, aE = D, D, D, D

		new_quality_scent = np.zeros(shape=self.size)

		for x in xrange(1, self.x_max):
			for y in xrange(1, self.y_max):
				new_quality_scent[x, y] = aN*self.quality_scent[x, y+1] + aS*self.quality_scent[x, y-1] + aE*self.quality_scent[x+1, y] + aW*self.quality_scent[x-1, y] 	+ (aP0 - (aN+aS+aW+aS-SP)) * self.quality_scent[x, y]

		self.quality_scent = new_quality_scent / aP


	def plot_quality_basic(self, fig):
		im = plt.imshow(self.quality_basic.T, cmap=plt.cm.Greens)

	def plot_quality_scent(self, fig):
		im = plt.imshow(self.quality_scent.T, cmap=plt.cm.Greens)

	def plot_quality_global(self, fig):
		im = plt.imshow(self.quality_global.T, cmap=plt.cm.Greens)


class Animal(object):
	'''
	Animal with position in environment
	'''
	def __init__(self, environment):
		self.position = np.array(environment.centre, dtype=int) 
		self.x = self.position[0]
		self.y = self.position[1]
		self.position_history = self.position
		self.subjective_environmnet = Environment(environment.size)

	def move(self, environment, m=1):
		'''
		Move the animal based on the quiality of its environment
		'''

		# get adjacent qualities
		adjacent_positions = [self.position + move for move in moves]
		adjacent_qualities = [environment.quality_global[tuple(p)]**m for p in adjacent_positions]
		move_index = weighted_choice(adjacent_qualities)

		# update animal position 
		self.position = self.position + moves[move_index]
		self.x, self.y = tuple(self.position)
		self.position_history = np.vstack((self.position_history, self.position))


	def plot_history(self, fig):
		'''
		Line of walk
		'''
		X = self.position_history[:, 0]
		Y = self.position_history[:, 1]
		plt.plot(X, Y, 'b', alpha=0.15, lw=0.5)

	def plot_location(self, fig):
		X, Y = tuple(self.position_history)
		plt.scatter(X, Y, c='r', s=50, marker='o', zorder=2000)

	def plot_history_scatter(self, fig):
		'''
		Scatter plot of previous positions
		'''
		X = self.position_history[:, 0]
		Y = self.position_history[:, 1]
		plt.scatter(X, Y, c="b", s=10, alpha=0.05, marker="s")

