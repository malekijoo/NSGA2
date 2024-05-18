import numpy as np
from NSGAII import Nsgaii
import matplotlib.pyplot as plt


# Initialize Algorithm
alg = Nsgaii(
    determination = 100,
    pop_size = 100,
    p_crossover = 0.7,
    alpha = 0.1,
    p_mutation = 0.3,
    mu = 0.05,
    verbose = True,
)

# Solve the Problem
results = alg.run()
pop = results['population']
F = results['F']


# Plot Results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
pf_costs = np.array([pop[i].objectives for i in F[0]])
ax.scatter(pf_costs[:, 0], pf_costs[:, 1], pf_costs[:, 2],  marker = 'o')
ax.set_xlabel('Cost')
ax.set_ylabel('CO2')
ax.set_zlabel('Social Factor')

plt.show()




