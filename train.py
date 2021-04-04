import numpy as np
from numpy.linalg import inv
from numpy import matmul
from PIL import Image
import sys
from utils.interpolation import mosaic, lin_interp, cubic_interp, get_samples, generate_B, fill_block, fill_missing, error
from utils.regression import mat_to_col, gen_weights


# using command terminal arguments
# input_filename = sys.argv[1]
# output_name = sys.argv[2]
# using hardcoded file path
# NOTE: if the commandline argument is failing just uncomment this and replace with the file path desired
input_filename = "data/in/raptors.jpg"
output_name = "data/in/raptors.png"



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

for x in range(0, img_data.shape[0], 5):
    for y in range(0, img_data.shape[0], 5):
        # GB PATCH
        if(x == 0):
            block = img_data[:5, y:y+5]
            col = mat_to_col(block)
            g = G[x+2][y+2]
            b = B[x+2][y+2]
            if(input_GB.shape == (1,0)):
                input_GB = col
                GB = [[g, b]]
            else:
                input_GB = np.append(input_GB, col, axis = 0)
                GB = np.append(GB, [[g, b]], axis = 0)
        # RB PATCH
        elif((x/5)%2 == 1):
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            r = R[x+2][y+2]
            b = B[x+2][y+2]
            if(input_RB.shape == (1,0)):
                input_RB = col
                RB = [[r, b]]
            else:
                input_RB = np.append(input_RB, col, axis = 0)
                RB = np.append(RB, [[r, b]], axis = 0)
        # RB_2 PATCH
        elif((x/5)%2 == 0):
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            r = R[x+2][y+2]
            b = B[x+2][y+2]
            if(input_RB_2.shape == (1,0)):
                input_RB_2 = col
                RB_2 = [[r, b]]
            else:
                input_RB_2 = np.append(input_RB_2, col, axis = 0)
                RB_2 = np.append(RB_2, [[r, b]], axis = 0)
        # RG PATCH
        # elif(x == img_data.shape[0]-5):
        else:
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            r = R[x+2][y+2]
            g = G[x+2][y+2]
            if(input_RG.shape == (1,0)):
                input_RG = col
                RG = [[r, g]]
            else:
                input_RG = np.append(input_RG, col, axis = 0)
                RG = np.append(RG, [[r, g]], axis = 0)



### Compute Coefficients and Store in txt file
f = open("weights.txt", "a")
## TYPE A GB
# print(GB.T, GB)
G_coeff, B_coeff = gen_weights(input_GB, GB.T)
f.write(np.array2string(G_coeff))
f.write(np.array2string(B_coeff))
## TYPE B RB
R_coeff, B_coeff = gen_weights(input_RB, RB.T)
f.write(np.array2string(R_coeff))
f.write(np.array2string(B_coeff))
## TYPE C RB
R_coeff, B_coeff = gen_weights(input_RB_2, RB_2.T)
f.write(np.array2string(R_coeff))
f.write(np.array2string(B_coeff))
# TYPE D RG
print(input_RG, RG)
R_coeff, G_coeff = gen_weights(input_RG, RG.T)
f.write(np.array2string(R_coeff))
f.write(np.array2string(G_coeff))

f.close()

