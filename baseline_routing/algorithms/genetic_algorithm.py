import logging
import numpy as np
from functools import reduce
from algorithms.random_search import RandomSearch
from solutions.solution import Solution
import utils as uls
import arcpy
from arcpy import env
from arcpy.sa import *
arcpy.env.workspace = r'C:\Users\Moritz\Desktop\Bk.gdb'
arcpy.env.outputZFlag = "Enabled"
arcpy.CheckOutExtension('Spatial')
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
arcpy.env.overwriteOutput = True
import sys
from time import gmtime, strftime
#arcpy.env.gpuId = 1

class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m, aimed_point_amount_factor):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.aimed_point_amount_factor = aimed_point_amount_factor

    def initialize(self):
        self.population = self._generate_random_valid_chromosomes()
        uls.non_dominated_sort(self.population)
        uls.crowding_distance(self.population)

    #params 3d space: x_y_limits, z_sigma, sample_point_distance
    #params ga: p_c, p_m, population_size, selection_pressure, n_point_crossover, percentage_disturb, max_disturbance, percentage_inserted_and_deleted, group_size_mutation

    def search(self, n_iterations,lgr, report=False, log=False, dplot=None):


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
            iterator_offspring_size = 0
            array_w_fcclassnames = [solution.PointFCName for solution in self.population]
            copy_parent_population = uls.copy_old_generation(self.population)
            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c:
                    p1.representation = uls.check_fields_x_y_z_gridz(p1, self.problem_instance.IDW)
                    p2.representation = uls.check_fields_x_y_z_gridz(p2, self.problem_instance.IDW)
                    off1, off2 = self._crossover(p1, p2)


                if self._random_state.uniform() < self.p_m:
                    off1.representation = uls.check_fields_x_y_z_gridz(off1, self.problem_instance.IDW)
                    off1 = self._mutation(off1)
                    off2.representation = uls.check_fields_x_y_z_gridz(off2, self.problem_instance.IDW)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    ##after mutation and crossover, only the representation of the new chromosome is kept. The feature class needs to be updated and the fitness calculated

                    off1.LineFCName = p1.LineFCName
                    #Update the feature class according to the stored representation values of the x and y positions in the numpy array. It is necessary to give it a new name
                    off1.PointFCName =  copy_parent_population[iterator_offspring_size].PointFCName.split('__', 1)[0] + "__" + str(iteration)

                    #off1.representation = np.hstack([off1.representation,uls.calculateSlopes_eucDist_bAngle(off1.PointFCName)])

                    #same for seccond offspring
                    off2.LineFCName = p2.LineFCName
                    # Update the feature class according to the stored representation values of the x and y positions in the numpy array. It is necessary to give it a new name
                    off2.PointFCName = copy_parent_population[iterator_offspring_size+1].PointFCName.split('__', 1)[
                                           0] + "__" + str(iteration)
                    #off2.representation = uls.equalize_and_repair_representation_and_fc(off2, self.problem_instance.IDW, self.problem_instance.endpoints, 3,self.problem_instance.restricted_airspace)
                    #off2 = uls.check_synchronization(off2)
                    #off2.representation = np.hstack([off2.representation, uls.calculateSlopes_eucDist_bAngle(off2.PointFCName)])

                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    array_w_fcclassnames.append(off1.PointFCName)
                    array_w_fcclassnames.append(off2.PointFCName)
                    offsprings.extend([off1, off2])
                    iterator_offspring_size =iterator_offspring_size + 2

            while len(offsprings) > len(self.population):
                offsprings.pop()


            self.population, unaccepted_solutions, non_dominated_solutions = self._nsga(offsprings)

            self.elite = self._get_elite()
            if report:
                self._verbose_reporter_inner(self.elite, iteration)

            if log:
                log_event = [iteration,self.elite[0].PointFCName, self.elite[0].fitness[0],self.elite[1].PointFCName, self.elite[1].fitness[1],self.elite[2].PointFCName, self.elite[2].fitness[2],
                            self._phenotypic_diversity_shift(offsprings), [[sol.PointFCName, sol.fitness] for sol in non_dominated_solutions]]
                lgr.info(','.join(list(map(str, log_event))))


            #Delete all feature classes that are not in the new population
            # if iteration >= 1:
            #     uls.delete_fc_from_old_generation(array_w_fcclassnames, [solution.PointFCName for solution in self.population])

            if dplot is not None:
                _iterative_plot()

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1, p2, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, chromosome):
        mutant = self.mutation(chromosome, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self):
        fitness_array = []
        for solution in self.population:
            fitness_array.append([solution.PointFCName, solution.fitness[0], solution.fitness[1], solution.fitness[2]])
        fitness_array = np.array(fitness_array).reshape(len(self.population), 4)
        shortest_flighttime = np.argmin(fitness_array[:, 1])
        lowest_energy = np.argmin(fitness_array[:, 2])
        lowest_added_noise = np.argmin(fitness_array[:, 3])
        return [self.population[shortest_flighttime],self.population[lowest_energy],self.population[lowest_added_noise]]


    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_chromosomes(self):
        #s = self._generate_random_valid_solution()
        chromosomes = np.array([self._generate_random_valid_solution() for _ in range(self.population_size)])
        return chromosomes

    def _nsga(self, offsprings):
        combined_population = []
        for sol in self.population:
            combined_population.append(sol)
        for sol in offsprings:
            combined_population.append(sol)
        uls.non_dominated_sort(combined_population)
        uls.crowding_distance(combined_population)
        accepted_solutions = []

        lowest_rank = 0
        # find lowest possible rank
        for solution in combined_population:
            if solution.rank > lowest_rank:
                lowest_rank = solution.rank
        # append solutions with highest ranks to accepted solutions until the population_size is reached
        for r in range(lowest_rank):
            for solution in combined_population:
                if solution.rank == r:
                    accepted_solutions.append(solution)
            # if more accepted solutions exist than the population size allows, remove the solutions of lowest rank with lowest crowding distance
            if len(accepted_solutions) > self.population_size:
                accepted_solutions = np.array(accepted_solutions)
                reevalution_on_crowding_distance = []
                for sol in accepted_solutions:
                    if sol.rank == r:
                        reevalution_on_crowding_distance.append(sol)
                uls.crowding_distance(reevalution_on_crowding_distance)
                sorting_list = []
                for sol in reevalution_on_crowding_distance:
                    sorting_list.append([sol.crowding_distance, sol])
                sorting_list = np.array(sorting_list)
                while accepted_solutions.shape[0] > self.population_size:
                    lowest_crowding_distance = np.argmin(sorting_list[:, 0])
                    try:
                        pos_to_delete_sorting_list = \
                        np.where(sorting_list[:, 0] == sorting_list[np.argmin(sorting_list[:, 0])][0])[0][0]
                        pos_to_delete_acccepted_list = \
                        np.where(accepted_solutions == sorting_list[pos_to_delete_sorting_list][1])[0][0]
                        sorting_list = np.delete(sorting_list, pos_to_delete_sorting_list, 0)
                        accepted_solutions = np.delete(accepted_solutions, pos_to_delete_acccepted_list, 0)
                    except:
                        print("hmm")
                accepted_solutions = accepted_solutions.tolist()
                break
            elif len(accepted_solutions) == self.population_size:
                break
        unaccepted_solutions = list(set(self.population).difference(accepted_solutions))
        non_dominated_solutions = []
        for sol in accepted_solutions:
            if sol.rank == 1:
                non_dominated_solutions.append(sol)

        return accepted_solutions, unaccepted_solutions, non_dominated_solutions