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

for x in range(0, img_data.shape[0], 5):
    for y in range(0, img_data.shape[0], 5):
        # GB PATCH
        if(x == 0):
            block = img_data[:5, y:y+5]
            col = mat_to_col(block)
            if(input_GB.shape == (1,0)):
                input_GB = col
            else:
                input_GB = np.append(input_GB, col, axis = 0)
            g = img_data_rgb[1][x+2][y+2]
            b = img_data_rgb[2][x+2][y+2]
            GB = np.append(GB, [[g, b]], axis = 0)
        # RB PATCH
        elif((x/5)%2 == 1):
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            if(input_RB.shape == (1,0)):
                input_RB = col
            else:
                input_RB = np.append(input_RB, col, axis = 0)
            r = img_data_rgb[0][x+2][y+2]
            b = img_data_rgb[2][x+2][y+2]
            RB = np.append(RB, [[r, b]], axis = 0)
        # RB_2 PATCH
        elif((x/5)%2 == 0):
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            if(input_RB_2.shape == (1,0)):
                input_GB = col
            else:
                input_RB_2 = np.append(input_RB_2, col, axis = 0)
            r = img_data_rgb[0][x+2][y+2]
            b = img_data_rgb[2][x+2][y+2]
            RB_2 = np.append(RB_2, [[r, b]], axis = 0)
        # RG PATCH
        else:
            block = img_data[x:x+5, y:y+5]
            col = mat_to_col(block)
            if(input_RG.shape == (1,0)):
                input_RG = col
            else:
                input_RG = np.append(input_RG, col, axis = 0)
            r = img_data_rgb[0][x+2][y+2]
            g = img_data_rgb[1][x+2][y+2]
            RG = np.append(RG, [[r, g]], axis = 0)



### Compute Coefficients and Store in txt file
f = open("weights.txt", "a")
G_A = gen_weights(input_GB, GB[0])
f.write(G_A)
B_A = gen_weights(input_GB, GB[1])
f.write(B_A)
R_A = gen_weights(input_RB, RB[0])
f.write(R_A)
B_A = gen_weights(input_RB, RB[1])
f.write(B_A)
R_A = gen_weights(input_RB_2, RB_2[0])
f.write(R_A)
B_A = gen_weights(input_RB_2, RB_2[1])
f.write(B_A)
R_A = gen_weights(input_RG, RG[0])
f.write(R_A)
G_A = gen_weights(input_RG, RG[0])
f.write(G_A)

f.close()

