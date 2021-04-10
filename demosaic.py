import numpy as np
from numpy.linalg import inv
from numpy import matmul
from PIL import Image
import sys
from utils.support import mosaic, pad_image, get_samples, error, bound
from utils.regression import mat_to_col
from numpy import loadtxt
import time


def demosaic(input_filename, output_name, patch_size=5):
    # load coefficients
    coeffs= loadtxt('weights.csv', delimiter=',')

    print("Opening {}".format(input_filename))
    img = Image.open(input_filename)
    img_data_rgb = np.asarray(img)

    ## get mosaic of image
    img_data = mosaic(img_data_rgb.T)

    # color channels
    r_data = get_samples(img_data, img_data.shape,"r", "data")
    g_data = get_samples(img_data, img_data.shape, "g", "data")
    b_data = get_samples(img_data, img_data.shape,"b", "data")

    img_width = img_data.shape[1]
    img_height = img_data.shape[0]

    pad = int(patch_size/2)
    n = pad
    img_padded = pad_image(img_data, pad)

    for x in range(int(img_data.shape[0])):
        for y in range(int(img_data.shape[1])):
            block = img_padded[(x+pad-n):(x+pad+n+1),(y+pad-n):(y+pad+n+1)]
            col = np.array([block.flatten()])
            # print(col.shape)
            if(x%2==0 and y%2==0):
                # PATCH TYPE A - GB
                g_data[x][y] = bound(np.sum(coeffs[0] * col[0]))
                b_data[x][y] = bound(np.sum(coeffs[1] * col[0]))
            elif(x%2==1 and y%2==0):
                # PATCH TYPE B - RB
                r_data[x][y] = bound(np.sum(coeffs[2] * col[0]))
                b_data[x][y] = bound(np.sum(coeffs[3] * col[0]))
            elif(x%2==0 and y%2==1):
                # PATCH TYPE C -RB
                r_data[x][y] = bound(np.sum(coeffs[4] * col[0]))
                b_data[x][y] = bound(np.sum(coeffs[5] * col[0]))
            elif(x%2==1 and y%2==1):
                # PATCH TYPE D -RG
                r_data[x][y] = bound(np.sum(coeffs[6] * col[0]))
                g_data[x][y] = bound(np.sum(coeffs[7] * col[0]))

    final_arr = np.asarray([r_data.T,g_data.T,b_data.T])

    c_img = Image.fromarray(np.uint8(final_arr.T),'RGB')
    filename = "demosaic-{}.png".format(output_name)
    c_img.save(output_name)
    print("Saved {}".format(output_name))

    print(error(img_data_rgb, final_arr))


def main():
    # using command terminal arguments
    input_filename = sys.argv[1]
    output_name = sys.argv[2]
    # using hardcoded file path
    # NOTE: if the commandline argument is failing just uncomment this and replace with the file path desired
    # input_filename = "data/in/{}"
    # output_name = "data/out/{}"

    # patch size 5 == 5x5 for example
    patch_size = 9
    demosaic(input_filename, output_name, patch_size)


if __name__ == "__main__":
    main()