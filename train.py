import numpy as np
from numpy.linalg import inv
from numpy import matmul
from PIL import Image
import sys
from utils.support import mosaic, pad_image
from utils.regression import mat_to_col, gen_weights
from numpy import savetxt
import time

# using command terminal arguments
# input_filename = sys.argv[1]
# output_name = sys.argv[2]
# using hardcoded file path
# NOTE: if the commandline argument is failing just uncomment this and replace with the file path desired
input_filename = "data/in/harden.jpg"
# output_name = "data/out/raptors.png"



print("Opening {}".format(input_filename))
img = Image.open(input_filename)
img_data_rgb = np.asarray(img)

## get mosaic of image
img_data = mosaic(img_data_rgb.T)

### PATCH TYPES GB, RB, RB_2, RG

GB = np.array([[]])
input_GB = np.array([[]])
RB = np.array([[]])
input_RB = np.array([[]])
RB_2 = np.array([[]])
input_RB_2 = np.array([[]])
RG = np.array([[]])
input_RG = np.array([[]])

### Pull patches into input Training sets and rgb values into output training sets
# print(img_data_rgb.T[0].T.shape)
R = img_data_rgb.T[0].T
G = img_data_rgb.T[1].T
B = img_data_rgb.T[2].T
img_width = img_data.shape[1]
img_height = img_data.shape[0]
print("Loading patches ... ")

pad = 2
img_padded = pad_image(img_data, pad)

start = time.time()
for x in range(int(img_data.shape[0])):
    for y in range(int(img_data.shape[1])):
        block = img_padded[(x+pad-2):(x+pad+3),(y+pad-2):(y+pad+3)]
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


print('{} train time'.format(time.time()-start))
### Compute Coefficients and Store in txt file
## TYPE A GB
print("Generating weights ... ")
coeff_a, coeff_b = gen_weights(input_GB, GB.T)
data_coefs = np.array([coeff_a, coeff_b])
## TYPE B RB
coeff_a, coeff_b = gen_weights(input_RB, RB.T)
data_coefs = np.append(data_coefs, [coeff_a, coeff_b], axis = 0)
## TYPE C RB
coeff_a, coeff_b = gen_weights(input_RB_2, RB_2.T)
data_coefs = np.append(data_coefs, [coeff_a, coeff_b], axis = 0)
# TYPE D RG
coeff_a, coeff_b = gen_weights(input_RG, RG.T)
data_coefs = np.append(data_coefs, [coeff_a, coeff_b], axis = 0)

# save to csv file
savetxt('weights.csv', data_coefs, delimiter=',')
print("weights.csv generated")




