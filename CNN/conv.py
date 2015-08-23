import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import pylab
from PIL import Image
import numpy

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

w_shp = (6, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')

b_shp = (6,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)


#######
# RUN #
#######

# open random image of dimensions 639x516
img = Image.open('/home/hadyelsahar/Desktop/img.jpg')
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 2063, 1599)
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 7, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 7, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 7, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(1, 7, 4); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.subplot(1, 7, 5); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(1, 7, 6); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.subplot(1, 7, 7); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.show()