import logging
import numpy as np
from functools import reduce
from random_search import RandomSearch
from solutions.solution import Solution


class PSO(RandomSearch):
    def __init__(self, problem_instance, random_state, swarm_size, social, cognitive, inertia):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.swarm_size = swarm_size
        self.social = social
        self.cognitive = cognitive
        self.inertia = inertia

    def initialize(self):
        self.swarm = self._generate_random_valid_particles()
        self.best_solution = self._get_current_best_particle(self.swarm)

    def search(self, n_iterations, report=False, log=False, dplot=None):
        if dplot is not None:
            dplot.background_plot(self.problem_instance.search_space, self.problem_instance.fitness_function)

            def _iterative_plot():
                points = np.array([particle.representation for particle in self.swarm])
                points = np.insert(points, points.shape[0], values=self.best_solution.representation, axis=0)
                points = np.vstack((points[:, 0], points[:, 1]))
                z = np.array([particle.fitness for particle in self.swarm])
                z = np.insert(z, z.shape[0], values=self.best_solution.fitness)
                dplot.iterative_plot(points, z, self.best_solution.fitness)

        for iteration in range(n_iterations):
            self._update_position()
            [self.problem_instance.evaluate(particle) for particle in self.swarm]
            self._update_lBest()
            self._update_gBest()

            if report:
                self._verbose_reporter_inner(self.best_solution, iteration)

            if dplot is not None:
                _iterative_plot()

    def _update_gBest(self):
        self.best_solution = self._get_best(self.best_solution, self._get_current_best_particle(self.swarm))

    def _update_position(self):
        for particle in self.swarm:
            r1 = self._random_state.uniform(size=self.problem_instance.dimensionality)
            social_factor = np.multiply(np.multiply(self.social, r1),
                                          (np.subtract(self.best_solution.representation, particle.representation)))
            r2 = self._random_state.uniform(size=self.problem_instance.dimensionality)
            cognitive_factor = np.multiply(np.multiply(self.cognitive, r2),
                                             (np.subtract(particle.lBest_representation, particle.representation)))
            particle.velocity = np.add(np.multiply(self.inertia, particle.velocity),
                                       np.add(cognitive_factor, social_factor))
            particle.representation = np.add(particle.representation, particle.velocity)

    def _update_lBest(self):
        for particle in self.swarm:
            if self.problem_instance.minimization:
                if particle.fitness <= particle.lBest_fitness:
                    particle.lBest_fitness = particle.fitness
                    particle.lBest_representation = particle.representation.copy()
            else:
                if particle.fitness >= particle.lBest_fitness:
                    particle.lBest_fitness = particle.fitness
                    particle.lBest_representation = particle.representation.copy()

    def _generate_random_valid_particles(self):
        particles = np.array([self._generate_random_valid_solution()
                              for _ in range(self.swarm_size)])
        # set lBest and initial velocity
        for particle_i in particles:
            particle_i.lBest_representation = particle_i.representation.copy()
            particle_i.lBest_fitness = particle_i.fitness
            particle_i.velocity = np.zeros(self.problem_instance.dimensionality)
        return particles

    def _get_current_best_particle(self, swarm):
        cBest_pointer = reduce(self._get_best, swarm)
        cBest = Solution(cBest_pointer .representation.copy())
        cBest.fitness = cBest_pointer .fitness
        cBest.valid = cBest_pointer .valid
        cBest.dimensionality = cBest_pointer .dimensionality

        return cBest