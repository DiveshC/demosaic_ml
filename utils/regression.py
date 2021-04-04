import numpy as np
from numpy.linalg import inv
from numpy import matmul
import sys

def mat_to_col(inp):
    mat_X = inp.shape[0]
    mat_Y = inp.shape[1]
    out = np.zeros((1, mat_Y*mat_X))
    inc = 0
    for x in range(mat_X):
        for y in range(mat_Y):
            out[0][inc] = inp[x][y]
            inc+=1

    # print(out)
    return out

def gen_weights(X, Y):
    print('Input X dimensions: {} Output Y dimensions: {}'.format(X.shape,Y.shape))
    x_inv = inv(matmul(X.T, X))
    x_plus = matmul(x_inv, X.T)
    
    A = matmul(x_plus, Y[0])
    B = matmul(x_plus, Y[1])
    return A,B
