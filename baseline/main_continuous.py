import utils as uls
from problems.continuous import Continuous
from algorithms.genetic_algorithm import GeneticAlgorithm

# setup problem
dplot = uls.Dplot()
hypercube = [(-5, 5), (-5, 5)]
problem_instance = Continuous(search_space=hypercube, fitness_function=uls.rastrigin)

# setup Genetic Algorithm
population_size = 10
p_c = .8
p_m = .2
n_iterations = 10

for seed in range(0, 1):
    # setup random state
    random_state = uls.get_random_state(seed)

    # execute Genetic Algorithm
    ga1 = GeneticAlgorithm(problem_instance=problem_instance, random_state=random_state,
                           population=population, selection=uls.parametrized_tournament_selection(0.2),
                           crossover=uls.one_point_crossover, p_c=p_c,
                           mutation=uls.parametrized_iterative_bit_flip(.5), p_m=p_m)
    ga1.initialize()
    ga1.search(n_iterations = n_iterations, report = True, log = False, dplot = dplot)