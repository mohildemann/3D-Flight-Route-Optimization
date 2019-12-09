import numpy as np
class SearchAlgorithm:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance


    def initialize(self):
        pass


    def search(self, n_iterations, report=False):
        pass


    def _get_best(self, candidate_a, candidate_b):
        if self.problem_instance.minimization:
            if candidate_a.fitness >= candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a
        else:
            if candidate_a.fitness <= candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a


    def verbose_reporter(self):
        print("Best solution found:")
        self.best_solution.print_()


    def _verbose_reporter_inner(self, elite, iteration):
        print("> > > Current best solutions at iteration %d:" % iteration)
        print("Shortest Flighttime: " + str(elite[0].fitness[0])+ "seconds. Fc: "+ elite[0].PointFCName)
        print("Lowest energy emmission: "  + str(elite[1].fitness[1])+ "kWh. Fc: "+ elite[1].PointFCName)
        print("Lowest added noise: " +  str(elite[2].fitness[2])+ "dB. Fc: "+elite[2].PointFCName)