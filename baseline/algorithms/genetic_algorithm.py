import logging
import numpy as np
from functools import reduce
from algorithms.random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m

    def initialize(self):
        self.population = self._generate_random_valid_chromosomes(self.population_size)
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False, log=False, dplot=None):
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        if dplot is not None:
            dplot.background_plot(self.problem_instance.search_space, self.problem_instance.fitness_function)

            def _iterative_plot():
                points = np.array([chromosome.representation for chromosome in self.population])
                points = np.insert(points, points.shape[0], values=self.best_solution.representation, axis=0)
                points = np.vstack((points[:, 0], points[:, 1]))
                z = np.array([chromosome.fitness for chromosome in self.population])
                z = np.insert(z, z.shape[0], values=self.best_solution.fitness)
                dplot.iterative_plot(points, z, self.best_solution.fitness)

        for iteration in range(n_iterations):
            offsprings = []

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            elite_offspring = self._get_elite(offsprings)
            self.best_solution = self._get_best(self.best_solution, elite_offspring)

            if report:
                self._verbose_reporter_inner(self.best_solution, iteration)

            if log:
                log_event = [iteration, self.best_solution.fitness, self.best_solution.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            # replacement
            if self.best_solution == elite_offspring:
                self.population = offsprings
            else:
                self.population = offsprings
                index = self._random_state.randint(self.population_size)
                self.population[index] = self.best_solution

            if dplot is not None:
                _iterative_plot()

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, chromosome):
        mutant = self.mutation(chromosome.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_chromosomes(self):
        chromosomes = np.array([self._generate_random_valid_solution()
                              for _ in range(self.population_size)])
        return chromosomes