import numpy as np
from copy import deepcopy
from population import population, init_pop
from problem import ThisProblem
import utils as ut


class Nsgaii:
    """
    A class to implement the NSGA-II multi-objective optimization algorithm
    Written By Amir H. Malekijoo
    Inspired by:
    Mostapha Kalami Heris, NSGA-II in Python (URL: https://yarpiz.com), Yarpiz, 2023.

    """

    def __init__(self, determination=100, pop_size=100, p_crossover=0.7, alpha=0, p_mutation=0.2, mu=0.02, verbose=True):

        """Constructor for the NSGA-II object"""
        self.determination = determination
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.alpha = alpha
        self.p_mutation = p_mutation
        self.mu = mu
        self.verbose = verbose
        self.problem = ThisProblem(n_obj=3, n_ie=13, n_e=4)


    def run(self):

        pop = population(pop_size=self.pop_size)
        pop = init_pop(pop)
        self.problem.calculate_obj_const(pop)


        # Number of Mutatnts and CrossOver Mate
        # n_mutation = int(self.p_mutation * len(pop))
        n_mutate = int(self.p_mutation * len(pop))
        n_crossover = 2*int(self.p_crossover * len(pop) / 2)


        pop, F = ut.non_dominated_sorting(pop)
        pop = ut.calc_crowding_distance(pop, F)
        # print('here pop1', pop[1].rank, pop[1].objectives)
        pop, F = ut.sort_population(pop)

        # main evolution iterations
        for iter in range(self.determination):

            # crossover and mutation
            popc = ut.crossover(pop, n_crossover)
            self.problem.calculate_obj_const(popc)
            popm = ut.mutation(pop, n_mutate)
            self.problem.calculate_obj_const(popm)

            # Create Merged Population
            pop = deepcopy(pop + popc + popm)

            pop, F = ut.non_dominated_sorting(pop)
            pop = ut.calc_crowding_distance(pop, F)
            pop, F = ut.sort_population(pop)
            pop, F = ut.truncate_population(pop, F)
            # Show Iteration Information
            if self.verbose:
                print(f'Iteration {iter + 1}: Number of Pareto Members = {len(F[0])}')


        # Pareto Front Population
        pareto_pop = deepcopy([pop[i] for i in F[0]])

        return {
            'population': pop,
            'F': F,
            'pareto': pareto_pop,
        }








        if self.verbose:
            print(f'Iteration {iter + 1}: Number of Pareto Members = {len(F[0])}')

if __name__ == "__main__":
    alg = Nsgaii(pop_size=10, determination=3)
    alg.run()
