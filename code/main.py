import random

from models import World, Environment, Animal

import numpy as np


def main():
    '''
    Runs a home range simulation.
    '''

    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    # Simulation parameters
    environment_size = 301 
    n_iter = 5000
    n_animals = 10
    m = 5 # quality exponent weight

    # Make a world object, which contains the animals
    world = World(environment_size, n_animals)

    # Initialise the environment
    world.environment.add_gaussian_quality()

    # Iterate the world
    world.update(kind='environemnt_quality', n_iter=n_iter, m=m)

    # Visualise the results
    world.static_plot(quality_global=True, kde=True, history=True)


if __name__ == '__main__':
    main()
