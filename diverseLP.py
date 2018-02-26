from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import csv
import pandas as pd

## Read in Data 


data = pd.read_csv("simulatedDataCorrelated.csv")
V = data.values
M = V.shape[0]
N = V.shape[1]
print("Total Number of Participants:", N)


Q = 10 # number of people per desired team
W_d = int(N/2) 
K_d = int(np.floor(W_d/ Q)) # number of diverse teams
Q_reg = int(0.8 *  Q)
Q_slack = int(0.2 * Q)
W_reg = int(0.8 * W_d)
W_slack = int(0.2 * W_d)


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


# add a dummy skill
## we need VIJ to be sorted by non slackers first, then slackers
V = np.append(V, [[0]*W_d], axis = 0)


print("Generating ", K_d, "diverse teams")
# determine the objective function, coefficients of the X_ijk which are just the V_ij's
my_obj = np.array([V for _ in range(K_d)])
my_obj = my_obj.flatten()
num_vars = W_d * K_d * (M + 1)
my_ub = [1.0] * num_vars
my_lb = [0.0] * num_vars
assert(len(my_ub) == len(my_obj))

# ## String with possible char values 'B','I','C','S','N'; set ctype(j) to 
# ## 'B', 'I','C', 'S', or 'N' to indicate that x(j) should be binary, general integer, 
# ## continuous, semi-continuous or semi-integer (respectively).
my_ctype = "I" * num_vars
# NOTE: in the nonzero populate function we don't need the column or row name
my_rhs = [1.0] * (int(N/2)  +  (M) * K_d) + [-Q_reg] * K_d + [-Q_slack] * K_d
my_sense = "E" * (int(N/2)) + "L" * ((M)* K_d + 2 * K_d)
flatten = lambda l: [item for sublist in l for item in sublist]
assert(len(my_rhs) == len(my_sense))
print(len(my_sense))

def printshape(l):
    print(np.array(l).shape)

def populatebynonzero(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense)
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

    rows1 = [[i] * ((M + 1)* K_d) for i in range(W_d)]
    # dummy skill not constrained here
    rows2 = [[W_d + rownum] * (W_d) for rownum in range((M)* K_d)]
    rows3 = [[W_d + (M)* K_d + rownum]  * (W_reg * (M+1)) for rownum in range(K_d)]
    rows4 = [[W_d + (M)* K_d + K_d + rownum]  * (W_slack * (M+1)) for rownum in range(K_d)]

    rows = [rows1, rows2, rows3, rows4]
    for r in rows:
        printshape(r)
    rows = flatten(flatten(rows))
    print(len(rows))


    cols1 = [[[int(N/2) * (M + 1) * k + int(N/2) * j + i for j in range(M + 1)] for k in range(K_d)] for i in range(int(N/2))]
    # dummy skill not constrained here
    cols2 = [[[(M + 1) * int(N/2) * k + int(N/2) * j + i for i in range(int(N/2))] for k in range(K_d)] for j in range(M)]
    cols3 = [[[(M + 1) * int(N/2) * k + int(N/2) * j + i for i in range(W_reg)] for j in range(M + 1)] for k in range(K_d)] 
    cols4 = [[[(M + 1) * int(N/2) * k + int(N/2) * j + i for i in range(W_reg, W_reg + W_slack)] for j in range(M + 1)] for k in range(K_d)] 

    cols = [cols1, cols2, cols3, cols4]
    for c in cols:
        printshape(c)
    cols = flatten(flatten(flatten(cols)))
    print(len(cols))

    assert(len(cols) == len(rows))
    print("Vals len:")
    vals = [1.0] * (W_d * (2*M + 1) * K_d) + [-1.0] * ((W_reg + W_slack) * (M+1) * K_d)
    print(len(vals))

    coeffs = zip(rows, cols, vals)
    prob.linear_constraints.set_coefficients(coeffs)
 	
    assert(len(rows) == len(vals))


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