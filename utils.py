import numpy as np
from copy import deepcopy
from individual import Individual
from itertools import chain

# import random


def init(arr, lb, ub):

    arr_type = arr.dtype
    if arr_type == np.int64:
        return np.random.randint(lb, high=ub + 1)

    elif arr_type == np.float64:
        return np.random.uniform(low=lb, high=ub + 1)

def op_cross(p1, p2):
    """Performs crossover between two parents"""

    new_p1 = Individual()
    new_p2 = Individual()
    # print('p1 \n', p1.X['NTIRtir'])
    # print('p2 \n', p2.X['NTIRtir'])
    #
    # print('new_p1 \n', new_p1.X['NTIRtir'])

    for key, _ in p1.X.items():

        crossover_point = np.random.randint(0, p1.X[key].size)
        # if key == "NTIRtir":
        #     print('crossover_point ', crossover_point)
        _t1 = np.zeros(p1.X[key].size)
        _t2 = np.zeros(p2.X[key].size)

        _t1[:crossover_point] = deepcopy(p1.X[key].flatten()[:crossover_point])
        _t1[crossover_point:] = deepcopy(p2.X[key].flatten()[crossover_point:])

        _t2[crossover_point:] = deepcopy(p1.X[key].flatten()[crossover_point:])
        _t2[:crossover_point] = deepcopy(p2.X[key].flatten()[:crossover_point])

        new_p1.X[key] = deepcopy(_t1.reshape(p1.X[key].shape))
        new_p2.X[key] = deepcopy(_t2.reshape(p2.X[key].shape))

    # print('new_p1 \n', new_p1.X['NTIRtir'])
    # print('new_p2 \n', new_p2.X['NTIRtir'])
    return deepcopy(new_p1), deepcopy(new_p2)


def crossover(pop, n_crossover):
    # Crossover
    popc = [[None, None] for _ in range(n_crossover // 2)]
    # print(popc, (n_crossover // 2))
    for k in range(n_crossover // 2):
        parents = np.random.choice(range(len(pop)), size=2, replace=False)
        p1 = pop[parents[0]]
        p2 = pop[parents[1]]
        popc[k][0], popc[k][1] = op_cross(p1, p2)

    # Flatten Offsprings List
    popc = list(chain(*popc))

    return popc



def op_mutate(p):
    """ Performs mutation on an individual """

    mu = 0.02
    size = 0

    for i, v in p.X.items():
        size += v.size

    n_mu = np.ceil(mu * size)
    # print('size ', size, n_mu)
    y = deepcopy(p)
    J = np.random.choice(list(p.X.keys()), int(n_mu), replace=False)
    # print(J)

    for key in J:
        y_size = y.X[key].size
        # print(y.X[key])
        ith_val = np.random.randint(0, y_size)
        # print(key, ith_val)
        _temp = y.X[key].flatten()
        _temp[ith_val] = init(y.X[key], y.lb[key], y.ub[key])
        _temp = _temp.reshape(y.X[key].shape)
        y.X[key] = deepcopy(_temp)
        # print(y.X[key])
    return deepcopy(y)


def mutation(pop, n_mutation):
    """
    Mutation
    :param pop:
    :param n_mutation:
    :return: popm which is a list of mutated individuals

    """

    popm = [None for _ in range(n_mutation)]
    for k in range(n_mutation):
        p = pop[np.random.randint(len(pop))]
        popm[k] = op_mutate(p)
    return popm


def constraint_dominates(p, q):
    """ implemented from https://www.youtube.com/watch?v=k_3IKDUuM9E """
    if p.constraint_violation_count < q.constraint_violation_count:
        return True
    elif p.constraint_violation_count == q.constraint_violation_count:
        return dominates(p, q)
    else:
        False


def dominates(p, q):
    """Checks if p dominates q"""
    return all(p.objectives <= q.objectives) and any(p.objectives < q.objectives)


def non_dominated_sorting(pop):
    """Perform Non-dominated Sorting on a Population"""

    pop_size = len(pop)


    # Initialize Domination Stats
    domination_set = [[] for _ in range(pop_size)]
    dominated_count = [0 for _ in range(pop_size)]

    # Initialize Pareto Fronts
    F = [[]]

    # Find the first Pareto Front
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            # Check if i dominates j
            if constraint_dominates(pop[i], pop[j]):
                domination_set[i].append(j)
                dominated_count[j] += 1

            # Check if j dominates i
            elif constraint_dominates(pop[j], pop[i]):
                domination_set[j].append(i)
                dominated_count[i] += 1

        # If i is not dominated at all
        if dominated_count[i] == 0:
            pop[i].rank = 0
            F[0].append(i)

    # Pareto Counter
    k = 0

    while True:

        # Initialize the next Pareto front
        Q = []

        # Find the members of the next Pareto front
        for i in F[k]:
            for j in domination_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    pop[j].rank = k + 1
                    Q.append(j)

        # Check if the next Pareto front is empty
        if not Q:
            break

        # Append the next Pareto front
        F.append(Q)

        # Increment the Pareto counter
        k += 1

    return deepcopy(pop), F




def calc_crowding_distance(pop, F):
    """Calculate the crowding distance for a given population"""

    # Number of Pareto fronts (ranks)
    parto_count = len(F)

    # Number of Objective Functions
    n_obj = pop[0].objectives.size

    # Iterate over Pareto fronts
    for k in range(parto_count):
        costs = np.array([pop[i].objectives for i in F[k]])
        n = len(F[k])
        d = np.zeros((n, n_obj))

        # Iterate over objectives
        for j in range(n_obj):
            idx = np.argsort(costs[:, j])
            d[idx[0], j] = np.inf
            d[idx[-1], j] = np.inf

            for i in range(1, n - 1):
                d[idx[i], j] = costs[idx[i + 1], j] - costs[idx[i - 1], j]
                d[idx[i], j] /= costs[idx[-1], j] - costs[idx[0], j]

        # Calculate Crowding Distance
        for i in range(n):
            pop[F[k][i]].crowding_distance = sum(d[i, :])

    return deepcopy(pop)


def sort_population(pop):
    """Sorts a population based on rank (in asceding order) and crowding distance (in descending order)"""
    pop = sorted(pop, key=lambda x: (x.rank, -x.crowding_distance))

    max_rank = pop[-1].rank
    F = []
    for r in range(max_rank + 1):
        F.append([i for i in range(len(pop)) if pop[i].rank == r])

    return pop, F


def truncate_population(pop, F, pop_size=None):
    """Truncates a population to a given size"""

    if pop_size is None:
        pop_size = len(pop)

    if len(pop) <= pop_size:
        return pop, F

    # Truncate the population
    pop = pop[:pop_size]

    # Remove the extra members from the Pareto fronts
    for k in range(len(F)):
        F[k] = [i for i in F[k] if i < pop_size]

    return pop, F





if __name__ == "__main__":
    from population import population, init_pop
    pop = population(10)
    pop = init_pop(pop)
    n_crossover = 2 * int(0.7 * len(pop) / 2)
    popc = crossover(pop, n_crossover)

    p_mutation = 0.2
    n_mutate = int(p_mutation * len(pop))
    popm = mutation(pop, n_mutate)
    print(len(popc), popc)
    print(len(popm), popm)
















#
# class Operation:
#
#     def __init__(self, problem, name):
#         self.pop = problem
#         self.name = name
#
#     def _do(self):
#         pass
#
#     def __repr__(self):
#         return "Oporation: {} ".format(self.name)
#
#
# class MySampling(Operation):
#
#     def __repr__(self):
#         return super().__repr__() + ", Sampling "
#
#     def _do(self, problem, n_samples, **kwargs):
#
#         n_var = problem.n_var
#         X = np.full((n_samples, n_var), None, dtype=object)
#
#         for i in range(n_samples):
#             for j in range(n_var):
#                 if j <= 11:
#                     low, high = 0, 1
#                     X[i, j] = np.random.randint(low, high=high + 1)
#                 elif j <= 123:
#                     low, high = 0, 10 ** 7
#                     X[i, j] = np.random.randint(low, high=high + 1)
#                 else:
#                     low, high = 0, 10 ** 7
#                     X[i, j] = np.random.uniform(low=low, high=high + 1)
#
#         return X
#
#
# class MyCrossover(Operation):
#
#     def __repr__(self):
#         return super().__repr__() + ", Crossover "
#
#     def __init__(self):
#
#         # define the crossover: number of parents and number of offsprings
#         super().__init__(2, 2)
#
#     def do(self, problem, X, **kwargs):
#
#         # The input of has the following shape (n_parents, n_matings, n_var)
#         _, n_matings, n_var = X.shape
#
#         # The output owith the shape (n_offsprings, n_matings, n_var)
#         # Because there the number of parents and offsprings are equal it keeps the shape of X
#         Y = np.full_like(X, None, dtype=object)
#
#         # for each mating provided
#         for k in range(n_matings):
#
#             # get the first and the second parent
#             a, b = X[0, k, :], X[1, k, :]
#             # print('a type ', type(a), a.shape, a[0])
#             off_a, off_b = np.full_like(a, None), np.full_like(b, None)
#
#             for i in range(n_var):
#                 if np.random.random() < 0.5:
#                     off_a[i] = a[i]
#                     off_b[i] = b[i]
#                 else:
#                     off_a[i] = b[i]
#                     off_b[i] = a[i]
#
#             # join the character list and set the output
#             Y[0, k, :], Y[1, k, :] = off_a.copy(), off_b.copy()
#
#         return Y
#
#
# class MyMutation(Operation):
#
#     def __repr__(self):
#         return super().__repr__() + ", Mutated "
#
#     def __init__(self):
#         super().__init__()
#
#     def do(self, problem, X, **kwargs):
#
#         # for each individual
#         n_offsprings, n_var = X.shape
#         binary_list = np.arange(0, 11 + 1)
#         integer_list = np.arange(12, 123 + 1)
#         real_list = np.arange(123, 199)
#         n = 4
#         rnd_offsprings_list = np.random.choice(np.arange(n_offsprings), n)
#
#         for i in rnd_offsprings_list:
#
#             r = np.random.random()
#             # with a probabilty of 5% - change the order of characters
#             if r < 0.05:
#
#                 permut_bin = np.random.permutation(binary_list)
#                 permut_int = np.random.permutation(integer_list)
#                 permut_real = np.random.permutation(real_list)
#
#                 for indx, j in enumerate(permut_bin):
#                     X[i, indx] = X[i, j]
#                 for indx, j in enumerate(permut_int):
#                     X[i, indx] = X[i, j]
#                 for indx, j in enumerate(permut_real):
#                     X[i, indx] = X[i, j]
#
#             # also with a probabilty of 40% - change a character randomly
#             elif r < 0.45:
#                 rnd_bin_mut_list  = np.random.choice(binary_list, 1)
#                 rnd_int_mut_list  = np.random.choice(integer_list, 5)
#                 rnd_real_mut_list = np.random.choice(real_list, 5)
#
#                 for j in rnd_bin_mut_list:
#                     low, high = 0, 1
#                     X[i, j] = np.random.randint(low, high=high + 1)
#                 for j in rnd_int_mut_list:
#                     low, high = 0, 10 ** 7
#                     X[i, j] = np.random.randint(low, high=high + 1)
#                 for j in rnd_real_mut_list:
#                     low, high = 0, 10 ** 7
#                     X[i, j] = np.random.uniform(low=low, high=high + 1)
#
#         return X

