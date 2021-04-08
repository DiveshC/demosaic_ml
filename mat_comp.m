ref = imread('data/in/tree.png');

mosaiced = imread('data/in/bayer/tree.png');
demosaiced = demosaic(mosaiced,'rggb');
imshow(mosaiced);
imshow(demosaiced);

err = immse(demosaiced, ref)