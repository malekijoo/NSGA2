import numpy as np
from individual import Individual
from copy import deepcopy

# import problem as pr

def population(pop_size):
    pop = []
    for i in range(pop_size):
        _temp = Individual()
        pop.append(deepcopy(_temp))
    return pop


def init_random(arr, lb, ub):

    new_arr = np.zeros_like(arr, dtype=arr.dtype)
    arr_type = arr.dtype
    if len(arr.shape) == 2:
        for i, j in np.ndindex(arr.shape):
            if arr_type == np.int64:
                new_arr[i, j] = np.random.randint(lb, high=ub + 1)

            elif arr_type == np.float64:
                new_arr[i, j] = np.random.uniform(low=lb, high=ub + 1)
    elif len(arr.shape) == 1:
        for i in np.ndindex(arr.shape):
            if arr_type == np.int64:
                new_arr[i] = np.random.randint(lb, high=ub + 1)

            elif arr_type == np.float64:
                new_arr[i] = np.random.uniform(low=lb, high=ub + 1)

    return new_arr


def init_pop(pop):

    pop_size = len(pop)
    for i in range(pop_size):
        item = pop[i].X
        ub   = pop[i].ub
        lb   = pop[i].lb
        for key, value in item.items():
            new_arr = init_random(value, lb[key], ub[key])
            pop[i].X[key] = deepcopy(new_arr)

    return pop



if __name__ == '__main__':

    pop = population(30)



