ref = imread('data/in/raptors.jpg');

mosaiced = imread('data/in/test_1.png');
demosaiced = demosaic(mosaiced,'rggb');
imshow(mosaiced);
imshow(demosaiced);

%err = immse(demosaiced, ref)