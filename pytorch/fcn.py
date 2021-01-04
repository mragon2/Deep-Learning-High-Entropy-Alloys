import numpy as np
import torch
import torchvision

"""
The implemented deep learning (DL) model is a fully convolutional network (FCN) characherized by
a symmetrical structure. The encoder and the decoder have 3 blocks of convolutional layers with 3x3 filters.
Between the encoder and the decoder, a bridge of convolutional layer is present. Each convolutional block
incorporates an input Conv2D layer, a residual block of 3 Conv2D layers and an output Conv2D layer.

The number of output channels in the convolutional blocks are 32,64 and 128 rispectively, 
while the bridge has 256 output channels. The last convolutional layer of the network is the score layer, 
with 1x1 filters and 5 output channels.

Each convolutional layer includes Batch Normalization and PReLU activation function.

In the encoder, Max-Pooling layers are used to halve the size of the image, while in the decoder the original
image size is restored using Bilinear Upsampling.

Skip connections are used to connect the convolutional blocks in the encoder and 
the symmetrical convolutional blocks in the decoder.

"""


class FCN(torch.nn.Module):

    def __init__(self,channels = 32, output_channels = 5):

        super(FCN, self).__init__()

        self.channels = channels

        self.output_channels = output_channels

        self.prelu = torch.nn.PReLU(init=0.25)

        self.relu = torch.nn.ReLU()


        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2)

        self.upsampling2d = torch.nn.UpsamplingBilinear2d(scale_factor=2)


        self.batch_norm_block1 = torch.nn.BatchNorm2d(num_features=self.channels)

        self.batch_norm_block2 = torch.nn.BatchNorm2d(num_features=self.channels*2)

        self.batch_norm_block3 = torch.nn.BatchNorm2d(num_features=self.channels*4)

        self.batch_norm_bridge = torch.nn.BatchNorm2d(num_features=self.channels*8)




        self.conv2d_block1_input = torch.nn.Conv2d(in_channels = 1,
                                            out_channels= self.channels,
                                           kernel_size=3,
                                           padding=1)

        self.conv2d_block1 = torch.nn.Conv2d(in_channels=self.channels,
                                            out_channels=self.channels,
                                            kernel_size=3,
                                            padding=1)

        self.conv2d_block2_input = torch.nn.Conv2d(in_channels=self.channels,
                                            out_channels=self.channels*2,
                                            kernel_size=3,
                                            padding=1)

        self.conv2d_block2 = torch.nn.Conv2d(in_channels=self.channels*2,
                                            out_channels=self.channels*2,
                                            kernel_size=3,
                                            padding=1)


        self.conv2d_block3_input = torch.nn.Conv2d(in_channels=self.channels*2,
                                            out_channels=self.channels*4,
                                            kernel_size=3,
                                            padding=1)

        self.conv2d_block3 = torch.nn.Conv2d(in_channels=self.channels*4,
                                            out_channels=self.channels*4,
                                            kernel_size=3,
                                            padding=1)



        self.conv2d_bridge_input = torch.nn.Conv2d(in_channels=self.channels*4,
                                            out_channels=self.channels*8,
                                            kernel_size=3,
                                            padding=1)

        self.conv2d_bridge = torch.nn.Conv2d(in_channels=self.channels*8,
                                            out_channels=self.channels*8,
                                            kernel_size=3,
                                            padding=1)


        self.deconv2d_block3_input = torch.nn.Conv2d(in_channels=self.channels*8,
                                            out_channels=self.channels*4,
                                            kernel_size=3,
                                            padding=1)

        self.deconv2d_block3 = torch.nn.Conv2d(in_channels=self.channels*4,
                                            out_channels=self.channels*4,
                                            kernel_size=3,
                                            padding=1)

        self.deconv2d_block2_input = torch.nn.Conv2d(in_channels=self.channels*4,
                                            out_channels=self.channels*2,
                                            kernel_size=3,
                                            padding=1)

        self.deconv2d_block2 = torch.nn.Conv2d(in_channels=self.channels*2,
                                            out_channels=self.channels*2,
                                            kernel_size=3,
                                            padding=1)

        self.deconv2d_block1_input = torch.nn.Conv2d(in_channels=self.channels*2,
                                            out_channels=self.channels,
                                            kernel_size=3,
                                            padding=1)

        self.deconv2d_block1 = torch.nn.Conv2d(in_channels=self.channels,
                                            out_channels=self.channels,
                                            kernel_size=3,
                                            padding=1)

        self.score_layer = torch.nn.Conv2d(in_channels=self.channels,
                                            out_channels=self.output_channels,
                                            kernel_size=3,
                                            padding=1)



    def forward(self,x):

        down_sampling_block1_1 = self.conv2d_block1_input(x)

        down_sampling_block1_2 = self.prelu(self.batch_norm_block1(self.conv2d_block1(down_sampling_block1_1))) # 32,256,256

        down_sampling_block1_3 = self.prelu(self.batch_norm_block1(self.conv2d_block1(down_sampling_block1_2)))

        down_sampling_block1_4 = self.prelu(self.batch_norm_block1(self.conv2d_block1(down_sampling_block1_3))) + down_sampling_block1_1

        down_sampling_block1_5 = self.prelu(self.batch_norm_block1(self.conv2d_block1(down_sampling_block1_4)))

        pooling_block1 = self.maxpool2d(down_sampling_block1_5) # 32,128,128


        down_sampling_block2_1 = self.conv2d_block2_input(pooling_block1) # 64,128,128

        down_sampling_block2_2 = self.prelu(self.batch_norm_block2(self.conv2d_block2(down_sampling_block2_1)))

        down_sampling_block2_3 = self.prelu(self.batch_norm_block2(self.conv2d_block2(down_sampling_block2_2)))

        down_sampling_block2_4 = self.prelu(self.batch_norm_block2(self.conv2d_block2(down_sampling_block2_3))) + down_sampling_block2_1

        down_sampling_block2_5 = self.prelu(self.batch_norm_block2(self.conv2d_block2(down_sampling_block2_4)))

        pooling_block2 = self.maxpool2d(down_sampling_block2_5) # 64,64,64


        down_sampling_block3_1 = self.conv2d_block3_input(pooling_block2) # 128,64,64

        down_sampling_block3_2 = self.prelu(self.batch_norm_block3(self.conv2d_block3(down_sampling_block3_1)))

        down_sampling_block3_3 = self.prelu(self.batch_norm_block3(self.conv2d_block3(down_sampling_block3_2)))

        down_sampling_block3_4 = self.prelu(self.batch_norm_block3(self.conv2d_block3(down_sampling_block3_3))) + down_sampling_block3_1

        down_sampling_block3_5 = self.prelu(self.batch_norm_block3(self.conv2d_block3(down_sampling_block3_4)))

        pooling_block3 = self.maxpool2d(down_sampling_block3_5) # 128,32,32


        bridge_1 = self.conv2d_bridge_input(pooling_block3) # 256,32,32

        bridge_2 = self.prelu(self.batch_norm_bridge(self.conv2d_bridge(bridge_1)))

        bridge_3 = self.prelu(self.batch_norm_bridge(self.conv2d_bridge(bridge_2)))

        bridge_4 = self.prelu(self.batch_norm_bridge(self.conv2d_bridge(bridge_3))) + bridge_1

        bridge_5 = self.prelu(self.batch_norm_bridge(self.conv2d_bridge(bridge_4))) # 256,32,32


        up_sampling_block3 = self.upsampling2d(bridge_5) # 256,64,64

        up_sampling_block3_1 = self.deconv2d_block3_input(up_sampling_block3)

        up_sampling_block3_2 = self.prelu(self.batch_norm_block3(self.deconv2d_block3(up_sampling_block3_1 + down_sampling_block3_1 )))

        up_sampling_block3_3 = self.prelu(self.batch_norm_block3(self.deconv2d_block3(up_sampling_block3_2)))

        up_sampling_block3_4 = self.prelu(self.batch_norm_block3(self.deconv2d_block3(up_sampling_block3_3))) + up_sampling_block3_1

        up_sampling_block3_5 = self.prelu(self.batch_norm_block3(self.deconv2d_block3(up_sampling_block3_4))) # 128,64,64


        up_sampling_block2 = self.upsampling2d(up_sampling_block3_5) # 128,128,128

        up_sampling_block2_1 = self.deconv2d_block2_input(up_sampling_block2)

        up_sampling_block2_2 = self.prelu(self.batch_norm_block2(self.deconv2d_block2(up_sampling_block2_1 + down_sampling_block2_1)))

        up_sampling_block2_3 = self.prelu(self.batch_norm_block2(self.deconv2d_block2(up_sampling_block2_2)))

        up_sampling_block2_4 = self.prelu(self.batch_norm_block2(self.deconv2d_block2(up_sampling_block2_3))) + up_sampling_block2_1

        up_sampling_block2_5 = self.prelu(self.batch_norm_block2(self.deconv2d_block2(up_sampling_block2_4))) # 64,128,128



        up_sampling_block1 = self.upsampling2d(up_sampling_block2_5) # 64,256,256

        up_sampling_block1_1 = self.deconv2d_block1_input(up_sampling_block1)

        up_sampling_block1_2 = self.prelu(self.batch_norm_block1(self.deconv2d_block1(up_sampling_block1_1 + down_sampling_block1_1)))

        up_sampling_block1_3 = self.prelu(self.batch_norm_block1(self.deconv2d_block1(up_sampling_block1_2)))

        up_sampling_block1_4 = self.prelu(self.batch_norm_block1(self.deconv2d_block1(up_sampling_block1_3))) + up_sampling_block1_1

        up_sampling_block1_5 = self.prelu(self.batch_norm_block1(self.deconv2d_block1(up_sampling_block1_4))) # 32,256,256

        inference = self.relu(self.score_layer(up_sampling_block1_5)) # C,256,256

        return inference
