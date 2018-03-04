from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import itertools
import pandas as pd
# np.random.seed = 10

import heapq
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


from sys import argv
myargs = getopts(argv)
## -t is time limit, -p is p, -Q is number of people, -s is store values and make graphs


TIMELIMIT = int(myargs["-t"])
p = int(myargs["-p"])
Q = int(myargs["-Q"])# number of people per desired team
s = int(myargs["-s"])
k = int(myargs["-k"])
filename = str(myargs["-input"])
print(TIMELIMIT, p, Q)


data = pd.read_csv(filename, header = 0)
V = data.values
M = V.shape[0]
N = V.shape[1] - 1
print("Total Number of Participants:", N)

W_s = N
K_s = k # number of diverse teams
Q_reg = Q
Q_slack = 0
W_reg = W_s
W_slack = 0

print("Number of People: ", W_s)
print("Number of Teams: ", K_s)
print("Number of Skills: ", M)


### TODO randomly order the data, and mark participant's ID's, so that the output knows which person is which


# first column is the participant's ID
V = V[:, 1:]

# normalize the data if we are using real data, but in simulation, the standard deviation is already determined
# for j in range(M):
# 	V[j] = (V[j]- np.mean(V[j]))/np.std(V[j]) 
# 	print(np.mean(V[j]))
# 	print(np.std(V[j]))


G_js = [sum(heapq.nlargest(Q + 1, V[i])) for i in range(M)]
G = max(G_js)


print("G", G)
my_obj = np.array([1.0]*(K_s * M) + [0.0] * (K_s * M + W_s * K_s))
my_obj = my_obj.flatten()
num_vars = 2* K_s * M  + W_s * K_s 
my_ub = [G] * (K_s * M) + [1.0] * (K_s * M + W_s * K_s)
my_lb = [-cplex.infinity] * num_vars
# print("Number of Variables: ", num_vars)
assert(len(my_ub) == len(my_obj))

# G = Q


my_ctype = "C" * (K_s * M) + "B" * (K_s * M + W_s * K_s)
# NOTE: in the nonzero populate function we don't need the column or row name
## TODO RHS
my_rhs = [0] * (2 * K_s * M ) + [p]* K_s + [1] * W_s + [-Q_reg] * K_s + [Q_reg + 1] * K_s
my_sense = "L" * (2 * K_s * M) + "L" *  (K_s) + "E" * (W_s) + "L" * (K_s * 2)
flatten = lambda l: [item for sublist in l for item in sublist]
assert(len(my_rhs) == len(my_sense))
# print(len(my_sense))

def printshape(l):
    print(np.array(l).shape)


def populatebynonzero(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense)
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)
    # prob.variables.add(obj=my_obj,  ub=my_ub, types=my_ctype)

    rows1 = [[i] * (2) for i in range(K_s * M)]
    rows2 = [[[K_s * M + k*M + j] * (1 + W_s) for j in range(M)] for k in range(K_s)]
    rows2 = flatten(rows2)
    rows3 = [[K_s * M * 2 + rownum]  * (M) for rownum in range(K_s)]
    rows4 = [[K_s * M * 2 + K_s + rownum] * (K_s) for rownum in range(W_s)]
    rows5 = [[K_s * M * 2 + K_s + W_s + rownum] * (W_reg) for rownum in range(K_s)]
    ## no team too big
    rows6 = [[K_s * M * 2 + K_s + W_s + K_s + rownum] * (W_reg) for rownum in range(K_s)]

    rows = [rows1, rows2, rows3, rows4, rows5, rows6]
    # print("ROWSSHAPE")
    # for r in rows:
    #     printshape(r)
    
    rows = flatten(flatten(rows))


    cols1 = [[index, index + K_s * M] for index in range(K_s * M)] 
    cols2 = [[[k * M + j] + [K_s * M * 2 + k * W_s + i for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    cols2 = flatten(cols2)
    cols3 = [[K_s * M + k * M + j for j in range(M)] for k in range(K_s)]
    cols4 = [[K_s * M  * 2 + k * W_s + i for k in range(K_s)] for i in range(W_s)]
    cols5 = [[K_s * M  * 2 + k * W_s + i for i in range(W_reg)] for k in range(K_s)]
    cols6 = [[K_s * M  * 2 + k * W_s + i for i in range(W_reg)] for k in range(K_s)]

    cols = [cols1, cols2, cols3, cols4, cols5, cols6]
    # print("COLSHAPES")
    # for c in cols:
    #     printshape(c)
    cols = flatten(flatten(cols))

    assert(len(cols) == len(rows))

    rowcol = list(zip(rows,cols))
    # print("VALSHAPES")
    vals1 = [1.0, -G] * (M * K_s) 
    # printshape(vals1)
    temp = [[[1.0] + [-V[j][i] for i in range(W_s)] for j in range(M)] for k in range(K_s)]
    vals2 = flatten(flatten(temp))
    # printshape(vals2)
    vals3 = [1.0] * (K_s * M + K_s * W_s) + [-1.0] * (K_s * W_reg) + [1.0] * (K_s * W_reg)

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
print("Objective value  = ", my_prob.solution.get_objective_value())

numcols = my_prob.variables.get_num()
# print("Number of Variables", numcols)
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()


## s is whether to store values or not 
if s == 1:
    ### -slack + G is the team value, nonzero value is the skill that is assigned to that team
    team_values = np.array(slack[:K_s * M])
    team_values = team_values.reshape(K_s, M)
    # print([(idx,x) for idx, x in enumerate(team_values)])
    team_values_out = [[idx, -sum(team) + G, np.nonzero(team)[0]] for idx, team in enumerate(team_values)]

    df_team_values = pd.DataFrame(team_values_out, columns = ["Team Number", "Skill Total", "Skill Number"])
    df_team_values.to_csv("IndspecializedTeamValues" + str(N) + ".csv")

if s == 1:
    team_assigns = x[K_s * M * 2:]
    team_assigns = np.array(team_assigns)
    team_assigns = team_assigns.reshape(K_s, W_s)


if s ==1:
    nonzeros = []
    for i in range(K_s):
        for j in range(W_s):
            if team_assigns[i,j] >= 0.5:
                nonzeros.append([i,j])
    assert(len(nonzeros) == W_s)
    a = [b[1] for b in nonzeros]
    assert(len(set(a)) == len(nonzeros))

if s ==1:
    df_team_assigns = pd.DataFrame(nonzeros, columns = ["Team Number", "User ID"])
    ## NOTE THAT all indices must add # of diverse participants
    df_team_assigns.to_csv(str(filename) + "specializedAssignments" + str(N) + ".csv")

    ## dictionary of team assignments
    team_assigns_dict = {}
    for k,i in nonzeros:
        if k in team_assigns_dict:
            team_assigns_dict[k].append(i)
        else:
            team_assigns_dict[k] = [i]

    XKJ = x[K_s * M: 2 * K_s * M]
    XKJ = np.array(XKJ).reshape(K_s, M)


if s ==1:
    # credited skills
    for team_id in range(K_s):
        matrices = []
        for k,team in enumerate(XKJ):
            team_members = team_assigns_dict[k]
            matrix = []
            for j, skill in enumerate(team):
                matrix.append([(V[j,i] if skill == 1 else 0) for i in team_members])
            matrices.append(matrix)
        ax = sns.heatmap(matrices[team_id], annot = True)
        plt.ylabel("Skills")
        title = "A" + str(filename) + str(W_s) +"Specialized_Selected" + str(team_id) + "p=" + str(p) + ".png"
        plt.title(title)
        plt.savefig(title)
        plt.clf()

if s == 1:
    for team_id in range(K_s):
        # all skills
        matrices_all = []
        for k,team in enumerate(XKJ):
            team_members = team_assigns_dict[k]
            matrix = []
            for j, skill in enumerate(team):
                matrix.append([(V[j,i]) for i in team_members])
            matrices_all.append(matrix)
        ax = sns.heatmap(matrices_all[team_id], annot = True)
        plt.ylabel("Skills")
        title = "A" + str(filename) + str(W_s) + "Specialized_All" + str(team_id) + "p=" + str(p) + ".png"
        plt.title(title)
        plt.savefig(title)
        plt.clf()


    ##### Save as pkl then read in the other file!! :) 



