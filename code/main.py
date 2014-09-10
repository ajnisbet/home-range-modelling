import random
import numpy as np

from models import World, Environment, Animal


def main():
	'''
	Runs a home range simulation.
	'''

	# Set a seed, for repeatable results.
	seed = 1
	random.seed(seed)
	np.random.seed(seed)

	# Simulation parameters
	environment_size = 301 
	n_iter = 5000
	n_animals = 10
	m = 50 # quality exponent weight

	# Make a world object, which contains the animals
	world = World(environment_size, n_animals)

	# Initialise the environment with a gaussian quality distribution
	world.environment.add_gaussian_quality()

	# Iterate the world
	world.update(kind='environemnt_quality', n_iter=n_iter, m=m)

	# Visualise the results, with environment quality, and the history of animal movementy
	world.static_plot(quality_global=True, kde=True, history=True)


if __name__ == '__main__':
	main()
