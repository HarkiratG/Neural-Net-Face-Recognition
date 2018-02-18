# Neual-Net-Face-Recogition

Once you get your basic network running, experiment with preprocessing of the input images and the network architecture to see how this affects the performance.
● Image preprocessing
○ Use only simple preprocessing operations such as smoothing, contrast
enhancement, normalization, etc. It generally does not make sense to use complex preprocessing techniques such as the Fourier Transform in this context. The input to the network should still be images of the faces (possibly preprocessed) at their original resolution.
○ Implement preprocessing using the ReadFcn attribute of the imageDatastore class
   
■ imagedatastore reference:
https://www.mathworks.com/help/matlab/ref/imagedatastore.html
● Different network architectures
○ Numbers and types of layers
○ For each 2D convolution layer, the size of kernel, number of filters, padding,
stride, etc.
○ Max pooling
● Different training settings
○ Regularization
○ Learning rate (can be dynamic too)
○ Training options reference material is available here:
■ https://www.mathworks.com/help/nnet/ref/trainingoptions.html
