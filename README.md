# Inception-Resnet-V2

mplementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. The paper on these architectures is available at "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".

The models are plotted and shown in the architecture sub folder. Due to lack of suitable training data (ILSVR 2015 dataset) and limited GPU processing power, the weights are not provided.
-------

The python script 'inception_resnet_v2.py' contains the methods necessary to create the Inception ResNet v2 network. It is to be noted that scaling of the residuals is turned ON by default.

There are a few differences in the v2 network from the original paper:

[1] In the B blocks: 'ir_conv' nb of filters is given as 1154 in the paper, however input size is 1152.
This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size is reduced to 1152 to match input size.

[2] In the C blocks: 'ir_conv' nb of filter is given as 2048 in the paper, however input size is 2144.

This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size is increased to 2144 to match input size.

Usage:

from inception_resnet_v2 import create_inception_resnet_v2

model = create_inception_resnet_v2(scale=True)
------------

