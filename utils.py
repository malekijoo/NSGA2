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

    new_p1 = Individual()
    new_p2 = Individual()

    for key, _ in p1.X.items():

        crossover_point = np.random.randint(0, p1.X[key].size)

        _t1 = np.zeros(p1.X[key].size)
        _t2 = np.zeros(p2.X[key].size)

        _t1[:crossover_point] = deepcopy(p1.X[key].flatten()[:crossover_point])
        _t1[crossover_point:] = deepcopy(p2.X[key].flatten()[crossover_point:])

        _t2[crossover_point:] = deepcopy(p1.X[key].flatten()[crossover_point:])
        _t2[:crossover_point] = deepcopy(p2.X[key].flatten()[:crossover_point])

        new_p1.X[key] = deepcopy(_t1.reshape(p1.X[key].shape))
        new_p2.X[key] = deepcopy(_t2.reshape(p2.X[key].shape))

    return deepcopy(new_p1), deepcopy(new_p2)


def crossover(pop, n_crossover):
    # Crossover
    popc = [[None, None] for _ in range(n_crossover // 2)]
    for k in range(n_crossover // 2):
        parents = np.random.choice(range(len(pop)), size=2, replace=False)
        p1 = pop[parents[0]]
        p2 = pop[parents[1]]
        popc[k][0], popc[k][1] = op_cross(p1, p2)

    popc = list(chain(*popc))

    return popc



def op_mutate(p):
    """ Performs mutation on an individual """

    mu = 0.02
    size = 0

    for i, v in p.X.items():
        size += v.size

    n_mu = np.ceil(mu * size)
    y = deepcopy(p)
    J = np.random.choice(list(p.X.keys()), int(n_mu), replace=False)

    for key in J:
        y_size = y.X[key].size
        ith_val = np.random.randint(0, y_size)
        _temp = y.X[key].flatten()
        _temp[ith_val] = init(y.X[key], y.lb[key], y.ub[key])
        _temp = _temp.reshape(y.X[key].shape)
        y.X[key] = deepcopy(_temp)
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
    if p.constraint_violation_count < q.constraint_violation_count:
        return True
    elif p.constraint_violation_count == q.constraint_violation_count:
        return dominates(p, q)
    else:
        False


def dominates(p, q):
    return all(p.objectives <= q.objectives) and any(p.objectives < q.objectives)


def non_dominated_sorting(pop):

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

    pop = sorted(pop, key=lambda x: (x.rank, -x.crowding_distance))

    max_rank = pop[-1].rank
    F = []
    for r in range(max_rank + 1):
        F.append([i for i in range(len(pop)) if pop[i].rank == r])

    return pop, F


def truncate_population(pop, F, pop_size=None):

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

