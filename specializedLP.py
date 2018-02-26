from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import itertools
import pandas as pd
np.random.seed = 10

import heapq
# import time

# start = time.time()
TIMELIMIT = 100 

data = pd.read_csv("simulatedDataCorrelated200.csv", header = 0)
V = data.values
M = V.shape[0]
N = V.shape[1] - 1
print("Total Number of Participants:", N)


Q = 10 # number of people per desired team
W_d = int(N/2) 
K_d = int(np.floor(W_d/ Q)) # number of diverse teams
Q_reg = int(0.8 *  Q)
Q_slack = int(0.2 * Q)
W_reg = int(0.8 * W_d)
W_slack = int(0.2 * W_d)

W_s = N - W_d
K_s = int(np.floor((N - W_d)/Q))# number of specialized teams
print("Number of People: ", W_s)
print("Number of Teams: ", K_s)
print("Number of Skills: ", M)


### TODO randomly order the data, and mark participant's ID's, so that the output knows which person is which


# first column is the participant's ID
V = V[:, 1:]
# Take the second half for specialized LP
V = V[:, W_d:]

print(V.shape)


# normalize the data if we are using real data, but in simulation, the standard deviation is already determined
# for j in range(M):
# 	V[j] = (V[j]- np.mean(V[j]))/np.std(V[j]) 
# 	print(np.mean(V[j]))
# 	print(np.std(V[j]))

# determine the objective function, coefficients of the X_ijk which are just the V_ij's
my_obj = np.array([1.0]*(K_s * M) + [0.0] * (K_s * M + W_s * K_s))
my_obj = my_obj.flatten()
num_vars = 2* K_s * M  + W_s * K_s 
my_ub = [cplex.infinity] * (K_s * M) + [1.0] * (K_s * M + W_s * K_s)
my_lb = [0.0] * num_vars
# print("Number of Variables: ", num_vars)
assert(len(my_ub) == len(my_obj))

G = max([sum(heapq.nlargest(Q, V[i])) for i in range(M)])

print(G)
# G = Q


my_ctype = "C" * (K_s * M) + "B" * (K_s * M + W_s * K_s)
# NOTE: in the nonzero populate function we don't need the column or row name
## TODO RHS
my_rhs = [0] * (2 * K_s * M ) + [1]* (W_s + K_s) + [-Q_reg] * K_s + [-Q_slack] * K_s
my_sense = "L" * (2 * K_s * M) + "E"*(W_s + K_s) + "L" * (K_s * 2)
flatten = lambda l: [item for sublist in l for item in sublist]
assert(len(my_rhs) == len(my_sense))
# print(len(my_sense))

def printshape(l):
    print(np.array(l).shape)


def populatebynonzero(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense)
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

    rows1 = [[i] * (2) for i in range(K_s * M)]
    rows2 = [[[K_s * M + k*M + j] * (1 + W_s) for j in range(M)] for k in range(K_s)]
    rows2 = flatten(rows2)
    rows3 = [[K_s * M * 2 + rownum]  * (M) for rownum in range(K_s)]
    rows4 = [[K_s * M * 2 + K_s + rownum] * (K_s) for rownum in range(W_s)]
    rows5 = [[K_s * M * 2 + K_s + W_s + rownum] * (W_reg) for rownum in range(K_s)]
    rows6 = [[K_s * M * 2 + K_s + W_s + K_s + rownum] * (W_slack) for rownum in range(K_s)]


    rows = [rows1, rows2, rows3, rows4, rows5, rows6]
    # for r in rows:
    #     printshape(r)
    
    rows = flatten(flatten(rows))


    cols1 = [[index, index + K_s * M] for index in range(K_s * M)] 
    cols2 = [[[k * M + j] + [K_s * M * 2 + k * W_s + i for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    cols2 = flatten(cols2)
    cols3 = [[K_s * M + k * M + j for j in range(M)] for k in range(K_s)]
    cols4 = [[K_s * M  * 2 + k * W_s + i for k in range(K_s)] for i in range(W_s)]
    cols5 = [[K_s * M  * 2 + k * W_s + i for i in range(W_reg)] for k in range(K_s)]
    cols6 = [[K_s * M  * 2 + k * W_s + i for i in range(W_reg, W_reg + W_slack)] for k in range(K_s)]

    cols = [cols1, cols2, cols3, cols4, cols5, cols6] 

    cols = flatten(flatten(cols))

    assert(len(cols) == len(rows))

    rowcol = list(zip(rows,cols))
    # # print(len(rowcol) != len(set(rowcol)))
    # seen = set([(50, 0), (51, 1), (52, 2), (53, 3), (54, 4)])
    # for idx,rc in enumerate(rowcol):
    #     if rc in seen:
    #         print(idx, rc)
    #     seen.add(rc)

    vals1 = [1.0, -G] * (M * K_s) 
    temp = [[[1.0] + [-V[j][i] for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    vals2 = flatten(flatten(temp))
    vals3 = [1.0] * (K_s * M + K_s * W_s) + [-1.0] * (K_s * (W_reg + W_slack))
    vals = flatten([vals1,vals2, vals3])
    assert(len(vals) == len(rows))
    # print(list(zip(rows, cols, vals)))
    prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
 	


my_prob = cplex.Cplex()
handle = populatebynonzero(my_prob)

my_prob.parameters.timelimit.set(TIMELIMIT)
my_prob.solve()

print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
# the following line prints the corresponding string
print(my_prob.solution.status[my_prob.solution.get_status()])
print("Solution value  = ", my_prob.solution.get_objective_value())

numcols = my_prob.variables.get_num()
print("Number of Variables", numcols)
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()

#### TODOTODOTODO
### slack + G
### we want to use Q_jk to find which W_kj to use then get value of first kj slack values,
### and add G to them to get value of the team for the skill that was chosen, also print which skill
team_values = np.array(slack[:K_s * M])
team_values = team_values.reshape(K_s, M)
team_values_out = np.transpose(np.nonzero(team_values))
df_team_values = pd.DataFrame(team_values_out)
df_team_values.to_csv("specializedTeamValues" + str(N) + ".csv")

# for j in range(numrows):
#     print("Row %d:  Slack = %10f" % (j, slack[j]))

team_assigns = x[K_s * M * 2:]
team_assigns = np.array(team_assigns)
team_assigns = team_assigns.reshape(W_s, K_s)
nonzeros = []
for i in range(W_s):
    for j in range(K_s):
        if team_assigns[i,j] >= 0.5:
            nonzeros.append([i,j])


# df_team_assigns = pd.DataFrame(team_assigns_out)
### NOTE THAT all indices must add # of diverse participants
# df_team_assigns.to_csv("specializedAssignments" + str(N) + ".csv")


# for assign in team_assigns:
#     print("Column %d:  Value = %10f" % (j, x[j]))


##### Save as pkl then read in the other file!! :) 




