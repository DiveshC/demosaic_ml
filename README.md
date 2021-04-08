# Linear Least Squares Demosaicing Approach
This python project takes "ml" approach, where we train use rgb images to generate some coeficients. Then using those same coefficients we can demosaicing other images. Here is are the libraries and steps needed to run the project:
- Pillow
- numpy

```
$ pip3 install pillow

$ pip3 install numpy 
``` 

## Running 
### Training
To train, within the main directory you may run the following. It requires 1 argument:
- input rgb image path (used as reference for training)
```
python train.py "<input file path>"
```
### Demosaicing 
Within the main directory you can run the following command in the terminal. It takes 2 arguments:
- input rgb image path 
- desired output image path
```
$ python demosaic.py "<input file path>" "<output file path>"
```
Example snipet:
```
python demosaic.py "data/in/raptors.jpg" "data/out/demosaic-raptors.png"
```

If there is some issues with passing the file paths in the terminal command, within the demosaic.py file on line 13 and 14 there are hardcoded file paths that can be replaced with the desired image paths. Comment the lines above it as well (lines 9 and 10).
```
# using command terminal arguments
input_filename = sys.argv[1]
output_name = sys.argv[2]
# using hardcoded file path
# NOTE: if the commandline argument is failing just uncomment this and replace with the file path desired
# input_filename = "data/in/raptors.jpg"
# output_name = "data/in/raptors.png"
```

### File structure
There are 2 main files, the demosaic.py and a train.py. There is also a utils directory containning custom helper functions. Heres a quick breakdown of the purpose of each:
- demosaic.py for the interpolation of data and generating output rgb image
- train.py loads reference image and preforms Least squares algorithm to generate coefficients
- utils/regression.py has some helper functions for generating coefs and also loading patches
- utils/support.py some more helper functions for basic operations

The utils functions are imported in demosaic.py.

### Full Documentation
The [full report](./docs/Report.pdf) on the algorithms used and design process is in the docs folder 