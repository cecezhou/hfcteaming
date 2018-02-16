from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import itertools


N = 200 # total number of participants
Q = 10 # number of people per desired team
M_1 = 0 # number of other types of skills (working together, fluid intelligence)
M_2 = 12 # number of topic specific skill dimensions
M = M_1 + M_2# total number of skills, used for diverse teams, topics only used for specific teams
W_d = int(N/2)
K_d = int(np.floor(W_d/ Q)) # number of diverse teams
# print(K_d)
W_s = N - W_d
K_s = int(np.floor((N - W_d)/Q))# number of specialized teams


## 500 people
## assume 15 skills as a reasonable upper bound
## assume team size 10
## i.i.d uniform 0 1, make correlations
## want confirmation that it scales, and get a qualitative sense of what it's doing
## have it take in a config file for number of people, and V_ij, if passed in, otherwise generate uniformly
## reincorporate the turker part.....


# M by W_s matrix, rows are skills
print("Generating Random Values for ")
V = np.random.uniform(0, 1, (M, W_s))

# add a dummy skill
# V = np.append(V, [[0]*W_d], axis = 0)

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
assert(len(my_ub) == len(my_obj))
G = Q


my_ctype = "C" * (K_s * M) + "B" * (K_s * M + W_s * K_s)
# NOTE: in the nonzero populate function we don't need the column or row name
## TODO RHS
my_rhs = [0] * (2 * K_s * M ) + [1]* (W_s + K_s) + [-Q] * K_s
my_sense = "L" * (2 * K_s * M) + "E"*(W_s + K_s) + "L" * K_s
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
    rows2 = [[K_s * M + rownum] * (1 + W_s) for rownum in range(K_s * M)]
    rows3 = [[K_s * M * 2 + rownum]  * (M) for rownum in range(K_s)]
    rows4 = [[K_s * M * 2 + K_s + rownum] * (K_s) for rownum in range(W_s)]
    rows5 = [[K_s * M * 2 + K_s + W_s + rownum] * (W_s) for rownum in range(K_s)]

    rows = [rows1, rows2, rows3, rows4, rows5]
    for r in rows:
        printshape(r)
    
    rows = flatten(flatten(rows))


    cols1 = [[index, index + K_s * M] for index in range(K_s * M)] 
    cols2 = [[[k * M + j] + [K_s * M * 2 + k * W_s + i for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    cols2 = flatten(cols2)
    cols3 = [[k * M + j for j in range(M)] for k in range(K_s)]
    cols4 = [[k * W_s + i for k in range(K_s)] for i in range(W_s)]
    cols5 = [[k * W_s + i for i in range(W_s)] for k in range(K_s)]
    cols = [cols1, cols2, cols3, cols4, cols5] 

    # for c in cols:
    #     printshape(c)

    cols = flatten(flatten(cols))

    # rowcol = list(zip(rows,cols))
    # # print(len(rowcol) != len(set(rowcol)))
    # seen = set([(50, 0), (51, 1), (52, 2), (53, 3), (54, 4)])
    # for idx,rc in enumerate(rowcol):
    #     if rc in seen:
    #         print(idx, rc)
    #     seen.add(rc)


    vals1 = [1.0, -G] * (M * K_s) 
    temp = [[[1.0] + [-V[j][i] for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    vals2 = flatten(flatten(temp))
    vals3 = [1.0] * (K_s * M + K_s * W_s) + [-1.0] * (K_s * W_s)
    vals = flatten([vals1,vals2, vals3])
    assert(len(vals) == len(rows))
    coeffs = zip(rows, cols, vals)

    prob.linear_constraints.set_coefficients(coeffs)
 	


my_prob = cplex.Cplex()
handle = populatebynonzero(my_prob)


print("done generating")


my_prob.solve()

print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
# the following line prints the corresponding string
print(my_prob.solution.status[my_prob.solution.get_status()])
print("Solution value  = ", my_prob.solution.get_objective_value())

numcols = my_prob.variables.get_num()
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()

# for j in range(numrows):
#     print("Row %d:  Slack = %10f" % (j, slack[j]))
for j in range(numcols):
    print("Column %d:  Value = %10f" % (j, x[j]))

##### Save as pkl then read in the other file!! :) 