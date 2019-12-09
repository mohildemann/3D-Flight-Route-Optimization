import utils as uls
from problems.continuous import Continuous
from algorithms.pso import PSO

# setup problem
dplot = uls.Dplot()
hypercube = [(-5, 5), (-5, 5)]
problem_instance = Continuous(search_space=hypercube, fitness_function=uls.rastrigin, minimization=True)

# setup PSO
social = 1.
cognitve = 1.
intertia = .1
swarm_size = 10
n_iter = 10

best = None
for seed in range(0, 1):
    # setup random state
    random_state = uls.get_random_state(seed)

    # execute Genetic Algorithm
    pso = PSO(problem_instance=problem_instance, random_state=random_state,
              swarm_size=swarm_size, social=social, cognitive=cognitve, inertia=0.5)
    pso.initialize()
    pso.search(n_iterations=n_iter, report=True, log=False, dplot=dplot)
    best = pso.best_solution