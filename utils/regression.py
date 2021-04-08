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
    # for x in range(mat_X):
    #     out[0]

    # print(out)
    return out

def gen_weights(X, Y):
    x_inv = inv(matmul(X.T, X))
    x_plus = matmul(x_inv, X.T)

    A = matmul(x_plus, Y[0])
    B = matmul(x_plus, Y[1])
    return A,B


def patchify(img, R, G, B, stopX, stopY, startX=0, startY=0, pad = 2):
    # actual RGB values of training image store in these arrays
    GB = np.array([[]])
    RB = np.array([[]])
    RB_2 = np.array([[]])
    RG = np.array([[]])


    # input samples from patch to train with
    input_GB = np.array([[]])
    input_RB = np.array([[]])
    input_RB_2 = np.array([[]])
    input_RG = np.array([[]])
    for x in range(startX, stopX):
        for y in range(startY, stopY):
            # block = img_padded[(x+pad-2):(x+pad+3),(y+pad-2):(y+pad+3)]
            block = img[(x+pad-2):(x+pad+3),(y+pad-2):(y+pad+3)]
            col = np.array([block.flatten()])
            # pull block and convert to column vector
            # GB PATCH (A)
            if(x%2==0 and y%2==0):
                g = G[x][y]
                b = B[x][y]
                if(input_GB.shape == (1,0)):
                    input_GB = col
                    GB = [[g, b]]
                else:
                    input_GB = np.append(input_GB, col, axis = 0)
                    GB = np.append(GB, [[g, b]], axis = 0)
            # RB PATCH (B)
            if(x%2==1 and y%2==0):
                r = R[x][y]
                b = B[x][y]
                if(input_RB.shape == (1,0)):
                    input_RB = col
                    RB = [[r, b]]
                else:
                    input_RB = np.append(input_RB, col, axis = 0)
                    RB = np.append(RB, [[r, b]], axis = 0)
            # RB_2 PATCH (C)
            if(x%2==0 and y%2==1):
                r = R[x][y]
                b = B[x][y]
                if(input_RB_2.shape == (1,0)):
                    input_RB_2 = col
                    RB_2 = [[r, b]]
                else:
                    input_RB_2 = np.append(input_RB_2, col, axis = 0)
                    RB_2 = np.append(RB_2, [[r, b]], axis = 0)
            # RG PATCH (D)
            if(x%2==1 and y%2==1):
                r = R[x][y]
                g = G[x][y]
                if(input_RG.shape == (1,0)):
                    input_RG = col
                    RG = [[r, g]]
                else:
                    input_RG = np.append(input_RG, col, axis = 0)
                    RG = np.append(RG, [[r, g]], axis = 0)
    return GB, RB, RB_2, RG, input_GB, input_RB, input_RB_2, input_RG

def append_img_slices(a_data, b_data):
    a, b, c, d, input_a, input_b, input_c, input_d = a_data
    a2, b2, c2, d2, input_a2, input_b2, input_c2, input_d2 = b_data
    a = np.append(a, a2, axis=0)
    b = np.append(b, b2, axis=0)
    c = np.append(c, c2, axis=0)
    d = np.append(d, d2, axis=0)
    input_a = np.append(input_a, input_a2, axis=0)
    input_b = np.append(input_b, input_b2, axis=0)
    input_c = np.append(input_c, input_c2, axis=0)
    input_d = np.append(input_d, input_d2, axis=0)
    return a, b, c, d, input_a, input_b, input_c, input_d