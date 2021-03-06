import numpy as np
from numpy.linalg import inv
from numpy import matmul
import sys

def mosaic(rgb_arr):
    mos = np.zeros(rgb_arr[0].T.shape)
    r = rgb_arr[0].T
    g = rgb_arr[1].T
    b = rgb_arr[2].T
    # R
    for i in range(0, r.shape[0], 2):
        for j in range(0, r.shape[1], 2):
            mos[i][j] = r[i][j]
    # G
    for i in range(g.shape[0]):
        init = 1-(i%2)
        for j in range(init, g.shape[1], 2):
            mos[i][j] = g[i][j]
    # B
    for i in range(1, b.shape[0], 2):
        for j in range(1, b.shape[1], 2):
            mos[i][j] = b[i][j]
    return mos



def get_samples(blk, dim, color="r", mode=None):
    f = np.zeros(dim)
    s = np.full(blk.shape, None)
    r = 0
    init = 0
    ## green samples
    if(color == "g"):
        k=0
        for i in range(blk.shape[0]):
            init = 1-(i%2)
            for j in range(init, blk.shape[1], 2):
                if(mode=="data"):
                    f[i][j] = blk[i][j]
                else:
                    f[k] = blk[i][j]
                k+=1
        return f
    ## red/blue samples
    if(color == "b"):
        init = 1
    for i in range(init, blk.shape[0], 2):
        c = 0
        for j in range(init, blk.shape[1], 2):
            f[r][c] = blk[i][j]
            s[i][j] = blk[i][j]
            c += 1
        r += 1
    
    if(mode == "data"):
        f=s
    return f


def fill_block(blk, B, F):
    for i in range(blk.shape[0]):
        for j in range(blk.shape[1]):
            if(blk[i][j] == None):
                blk[i][j] = cubic_interp(i, j, B, F)
    return blk     
'''
Args
blk - the block of data we do have samples for
coord - the x,y ranges for where the blk sits in the 8x8
desired_dim - desirable block size
replace - the value we fill in for the missing samples
'''
def fill_missing(blk,coord, desired_dim, replace):
    des = np.full(desired_dim, replace)
    des[coord[0]:coord[1], coord[2]:coord[3]] = blk[coord[0]:coord[1], coord[2]:coord[3]]
    return des

def bound(num):
    if(num>255):
        return 255
    elif(num<0):
        return 0
    else:
        return num

def error(inp, outp):
    ref = inp.T
    r = ref[0]
    g = ref[1]
    b = ref[2]
    rGen = outp[0]
    gGen = outp[1]
    bGen = outp[2]
    sumR=0
    sumG=0
    sumB=0
    Nsamples = r.shape[0]*r.shape[1]
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            sumR += (r[i][j] - rGen[i][j])**2
            sumG += (g[i][j] - gGen[i][j])**2
            sumB += (b[i][j] - bGen[i][j])**2
    
    return (sumR+sumG+sumB)/(Nsamples*3)


def pad_image(img, n):
    width = img.shape[1]
    height = img.shape[0]
    new_width = width + 2*n
    new_height = height + 2*n
    new_padded = np.zeros((height+2*n, width+2*n))
    new_padded[n:height+n, n:width+n] = img

    # top corner
    new_padded[0:n,0:n] = img[n][n]
    # right corner
    new_padded[0:n,new_width-n-1:new_width] = img[n][width-1]
    # bottom corner
    new_padded[new_height-n-1:new_height, 0:n] = img[height-1][n]
    # bottom right corner
    new_padded[new_height-n-1:new_height, new_width-n-1:new_width] = img[height-1][width-1]
    
    # pad top and bottom edges
    for y in range(n, width+n, n):
        # top edge
        new_padded[0:n, y: y+n] = img[n][y-n-1]
        # bottom edge
        new_padded[new_height-n-1:new_height, y: y+n] = img[height-1][y-n-1]

    # pad top and bottom edges
    for x in range(n, height+n, n):
        # left edge
        new_padded[x:x+n, 0:n] = img[x-n-1][n]
        # right edge
        new_padded[n:x+n, new_width-n-1: new_width] = img[x-n-1][width-1]

    return new_padded

