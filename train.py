import numpy as np
from numpy.linalg import inv
from numpy import matmul
from PIL import Image
import sys
from utils.support import mosaic, pad_image
from utils.regression import mat_to_col, gen_weights, patchify, append_img_slices
from numpy import savetxt
import time
import multiprocessing

# using command terminal arguments
input_filename = sys.argv[1]
# input_filename = 'data/in/{}'.format(input_filename)
# output_name = sys.argv[2]
# using hardcoded file path
# NOTE: if the commandline argument is failing just uncomment this and replace with the file path desired
# input_filename = "data/in/harden.jpg"
# output_name = "data/out/raptors.png"



print("Opening {}".format(input_filename))
img = Image.open(input_filename)
img_data_rgb = np.asarray(img)

## get mosaic of image
img_data = mosaic(img_data_rgb.T)
img_width = img_data.shape[1]
img_height = img_data.shape[0]

### Pull patches into input Training sets and rgb values into output training sets
# print(img_data_rgb.T[0].T.shape)
R = img_data_rgb.T[0].T
G = img_data_rgb.T[1].T
B = img_data_rgb.T[2].T

pad = 2
img_padded = pad_image(img_data, pad)

start = time.time()

# inorder to optimize the pulling of patches we split the image to 4 corners
# and then extracts all the patches from each corner and append them into the final training arrays
print("Loading patches ... ")
# top corners
corner_l_t = patchify(img_padded, R, G, B, int(img_height/2), int(img_width/2))
corner_r_t = patchify(img_padded, R, G, B, int(img_height/2), int(img_width), 0, int(img_width/2))  

# bottom corners
corner_l_b = patchify(img_padded, R, G, B, int(img_height), int(img_width/2), int(img_height/2))
corner_r_b = patchify(img_padded, R, G, B, int(img_height), int(img_width), int(img_height/2), int(img_width/2))    

# append quarters
top = append_img_slices(corner_l_t, corner_r_t)
bottom = append_img_slices(corner_l_b, corner_r_b)

# append halves
GB, RB, RB_2, RG, input_GB, input_RB, input_RB_2, input_RG = append_img_slices(top, bottom)
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

print('{} train time'.format(time.time()-start))


