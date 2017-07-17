"""
Train CNN with 3 conv_2d layers and 2 fcl layers

Pix2Code's CNN-encoder architecture:
(performs UFL by mapping an input image to a learned fixed-length vector):

Image (625x1250)
  --> 3x3 receptive fields convolved with stride 1 as in VGGNet (this done twice)
    --> fixed-size output vector
      --> CNN(width=32)
        --> CNN(width=64)
          --> CNN(width=128)
            --> 2 fully-connected layers (size=1024) applying the rectified linear unit activation
              --> output vector (to be concatenated with 1st LSTM output vector)
"""

from glimpse.encoders.cnn import CNN

cnn = CNN(conv_layers=3, fcl_layers=2)
cnn.train()