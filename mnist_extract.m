% Extract MNIST Images With MATLAB
% MATLAB code for extracting MNIST dataset images.

% MNIST Dataset:
% http://yann.lecun.com/exdb/mnist/

% Repository:
% https://github.com/MacwinWin/ExtractMNISTImagesWithMATLAB.git

imagestest = loadMNISTImages('t10k-images-idx3-ubyte');
labelstest = loadMNISTLabels('t10k-labels-idx1-ubyte');
imagestrain = loadMNISTImages('train-images-idx3-ubyte');
labelstrain = loadMNISTLabels('train-labels-idx1-ubyte');
[n1 m1]=size(imagestest);
[n2 m2]=size(imagestrain);
%imshow(reshape((imagestest(:,1)),28,28))
