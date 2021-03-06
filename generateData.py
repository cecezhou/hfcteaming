from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexError
import numpy as np
import random
import csv
import pandas as pd
from scipy.stats import percentileofscore

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


from sys import argv
myargs = getopts(argv)




M = 15 # number of skills
W = int(myargs["-W"]) # number of total people


# M by W_d matrix, rows are skills
# print("Generating Random Values Independently ")
V = np.random.uniform(0, 1, (M, W))
# print(V)

df = pd.DataFrame(V)
df.to_csv("simulatedDataIndependent" + str(W) + ".csv")


# print("Generating Random Values Correlated")
blocksize = 5
numblocks = int(M / blocksize)
rho = 0.7
covMatrix = np.zeros((M,M))
for i in range(numblocks):
	for j in range(blocksize):
		for k in range(blocksize):
			covMatrix[i * blocksize + j][i * blocksize + k] = rho
for i in range(M):
	covMatrix[i][i] = 1.0
# print("Covariance Matrix", covMatrix)

MVN = np.random.multivariate_normal(np.zeros(M), covMatrix, W)
MVN = MVN.T

for i in range(M):
	MVN[i] = [percentileofscore(MVN[i], x) for x in MVN[i]]

MVN = MVN/100
print(MVN.shape)
df = pd.DataFrame(MVN)
df.to_csv("simulatedDataCorrelated" + str(W) + ".csv")