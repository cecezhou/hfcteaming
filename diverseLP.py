from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import csv
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

## Read in Data 
data = pd.read_csv("simulatedDataCorrelated300.csv")
V = data.values
M = V.shape[0]
N = V.shape[1] - 1
print("Total Number of Participants:", N) ### optimal is 909 when there is time limit of 1000
TIMELIMIT = 1000

Q = 12 # number of people per desired team
W_d = int(N/2) 
K_d = int(np.floor(W_d/ Q)) # number of diverse teams
Q_reg = int(0.8 *  Q)
Q_slack = int(0.2 * Q)
W_reg = int(0.8 * W_d)
W_slack = int(0.2 * W_d)
s = 2
r = 2

### TODO randomly order the data, and mark participant's ID's, so that the output knows which person is which

# first column is the participant's ID
V = V[:, 1:]
# Take the first half for the diverse LP
V = V[:, 0:W_d]
print("Number of participants in Diverse Teams:", V.shape[1])
# normalize the data if we are using real data, but in simulation, the standard deviation is already determined
# for j in range(M):
#   V[j] = (V[j]- np.mean(V[j]))/np.std(V[j]) 
#   print(np.mean(V[j]))
#   print(np.std(V[j]))


# we need VIJ to be sorted by non slackers first, then slackers
# let W_reg and W_slack be the number of each respectively
# no longer need adding dummy skill // V = np.append(V, [[0]*W_d], axis = 0)

print("Generating ", K_d, "diverse teams")
# max_x  \sum_{i \in W_d} \sum_{j\in M} \sum_{k \in K_d} x_{ijk} v_{ij}
my_obj = np.array([V for _ in range(K_d)])
my_obj = my_obj.flatten()
my_obj = np.append(my_obj, [0.0] * (W_d * K_d))

num_vars = W_d * K_d * M + W_d * K_d
my_ub = [1.0] * num_vars
my_lb = [0.0] * num_vars
assert(len(my_ub) == len(my_obj))


my_ctype = "I" * num_vars
# RHS 
my_rhs = [1.0] * W_d + [s] * W_d + [0.0] * (W_d * M * K_d) + [r] * (M * K_d) + [-Q_reg] * K_d + [-Q_slack] * K_d
my_sense = "E" * W_d + "L" * (W_d  + W_d * M * K_d + M * K_d + 2 * K_d)
flatten = lambda l: [item for sublist in l for item in sublist]
assert(len(my_rhs) == len(my_sense))
print(len(my_sense))

def printshape(l):
    print(np.array(l).shape)

def populatebynonzero(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense)
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

    # \sum_{k\in K_d} x_{ik}=1, for all i\in W_d # {assign each person to a team}
    rows1 = [[i] * K_d for i in range(W_d)]
    # \sum_{j \in M} \sum_{k\in K_d} x_{ijk} <= s , for all i \in W_d # {assign each person to at most s sklls}
    rows2 = [[W_d + rownum] * (M * K_d) for rownum in range(W_d)]
    # x_{ijk} <= x_{ik}, for all i, all j, all k # {only credit a skill if i is on a team}
    rows3 = [[W_d * 2 + rownum] * 2 for rownum in range(W_d * M * K_d)]
    # \sum_{i \in U} x_{ijk} <= r, for all j \in M, for all k \in K {credit each real skill at most r times per team}
    rows4 = [[W_d * 2 + W_d * M * K_d + rownum] * (W_d) for rownum in range(M* K_d)]
    # \sum_{i \in U_reg} x_{ik} >= Q_reg, for all k \in K  {enough non-slackers per team}
    rows5 = [[W_d * 2 + W_d * M * K_d + M* K_d + rownum]  * (W_reg) for rownum in range(K_d)]
    # \sum_{i \in U_slack} x_{ik} >= Q_slack, for all k \in K  {enough slackers per team}
    rows6 = [[W_d * 2 + W_d * M * K_d + M* K_d + K_d + rownum]  * (W_slack) for rownum in range(K_d)]

    rows = [rows1, rows2, rows3, rows4, rows5, rows6]
    # for r in rows:
    #     printshape(r)
    rows = flatten(flatten(rows))
    # print(len(rows))

    # \sum_{k\in K} x_{ik}=1, for all i\in W_d {assign each person to a team}
    cols1 = [[W_d * M * K_d + W_d * k + i for k in range(K_d)] for i in range(W_d)]
    # \sum_{j \in M} \sum_{k\in K} x_{ijk} <= s , for all i \in W_d {assign each person to at most s sklls}
    cols2 = [[[W_d * M * k + W_d * j + i for k in range(K_d)] for j in range(M)] for i in range(W_d)]
    # x_{ijk} <= x_{ik}, for all i, all j, all k # {only credit a skill if i is on a team}
    # the ordering of these rows will be k,j,i if reshaped
    cols3 = [[[[W_d * M * k + W_d * j + i, W_d * M * K_d + W_d * k + i] for i in range(W_d)] for j in range(M)] for k in range(K_d)]
    # \sum_{i \in W_d} x_{ijk} <= r, for all j \in M, for all k \in K {credit each real skill at most r times per team}
    cols4 = [[[W_d * M * k + W_d * j + i for i in range(W_d)] for k in range(K_d)] for j in range(M)]
    # \sum_{i \in W_reg} x_{ik} >= Q_reg, for all k \in K  {enough non-slackers per team}
    cols5 = [[W_d * M * K_d + W_d * k + i for i in range(W_reg)] for k in range(K_d)]
    # \sum_{i \in W_slack} x_{ik} >= Q_slack, for all k \in K  {enough slackers per team}
    cols6 = [[W_d * M * K_d + W_d * k + i for i in range(W_reg, W_reg + W_slack)] for k in range(K_d)]

    cols2 = flatten(cols2); cols3 = flatten(flatten(cols3)); cols4 = flatten(cols4)
    cols = [cols1, cols2, cols3, cols4, cols5, cols6]
    # for c in cols:
    #     printshape(c)
    cols = flatten(flatten(cols))
    # print(len(cols))

    assert(len(cols) == len(rows))
    # print("Vals len:")
    vals = [1.0] * (K_d * W_d + K_d * M * W_d) + [1.0, -1.0] * (K_d * 
                W_d * M) + [1.0] * (W_d * K_d * M) + [-1.0] * ((W_reg + W_slack) * K_d)
    # print(len(vals))

    coeffs = zip(rows, cols, vals)
    prob.linear_constraints.set_coefficients(coeffs)
 	
    assert(len(rows) == len(vals))


my_prob = cplex.Cplex()
handle = populatebynonzero(my_prob)
my_prob.parameters.timelimit.set(TIMELIMIT)
my_prob.solve()

print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
# the following line prints the corresponding string
print(my_prob.solution.status[my_prob.solution.get_status()])
print("Solution value  = ", my_prob.solution.get_objective_value())

numcols = my_prob.variables.get_num()
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()

### Visualize Solution and Save Output
team_assigns = x[K_d * M * W_d:]
team_assigns = np.array(team_assigns).reshape(K_d, W_d)
nonzeros = []
for i in range(K_d):
    for j in range(W_d):
        if team_assigns[i,j] >= 0.5:
            nonzeros.append([i,j])

df_team_assigns = pd.DataFrame(nonzeros, columns = ["Team Number", "User ID"])
### NOTE THAT all indices must add # of diverse participants
df_team_assigns.to_csv("diverseAssignments" + str(N) + ".csv")
team_assigns_dict = {}
for k,i in nonzeros:
    if k in team_assigns_dict:
        team_assigns_dict[k].append(i)
    else:
        team_assigns_dict[k] = [i]


XIJK = x[:K_d * M * W_d]
XIJK = np.array(XIJK).reshape(K_d, M, W_d)

# credited skills
matrices = []
for k,team in enumerate(XIJK):
    team_members = team_assigns_dict[k]
    matrix = []
    for j, skill in enumerate(team):
        matrix.append([(V[j,i] if skill[i] != 0 else 0) for i in team_members])
    matrices.append(matrix)

# even skills that aren't credited
matrices_all = []
for k,team in enumerate(XIJK):
    team_members = team_assigns_dict[k]
    matrix = []
    for j, skill in enumerate(team):
        matrix.append([(V[j,i] if skill[i] != 0 else 0) for i in team_members])
    matrices_all.append(matrix)

ax = sns.heatmap(matrices[0], annot = True)
plt.show()
ax = sns.heatmap(matrices_all[0], annot = True)
plt.show()



## TOdo still need to look at specialized "value" to compare with specialized


##### Save as pkl then read in the other file!! :) 