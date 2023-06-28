# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:46:36 2022

@author: LIPS
"""

import torch
import torch.nn as nn
from pre_model import parse_cfg
from pre_model import model_initialization
from utilities import predict_transform
import numpy as np 

class Darknet(nn.Module):
    def __init__(self, config_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(config_file) # store the blocks in a new class attribute list
        self.net_info, self.module_list = model_initialization(self.blocks) # store the two separate sections (net info, convolutional layers) in two class attributes
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # as explained before, the first index belongs to the dark-net info and does not play a role in the forward pass
        outputs = {}  # We cache the outputs for the route layer for the route and shortcut layers
        write_output = 0 # This is to store the first scale detection tensor in the "detections" variable
        for module_index, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[module_index](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                # note: the first index will be in negative, which means we move to the left of the current layer index
                # BUT for the second index, we directly mention the layer number
                # For example, we have layers = -1, 36. the current index = 98, then
                if (layers[0]) > 0:
                    layers[0] = layers[0] - module_index # we subtract the current index module form the layers' number

                if len(layers) == 1:
                    x = outputs[module_index + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - module_index # Subtract the

                    # Concatenate the two layers
                    map1 = outputs[module_index + layers[0]]
                    map2 = outputs[module_index + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[module_index - 1] + outputs[module_index + from_] # it is always adding the "from_" layer output to the previous layer

            elif module_type == 'yolo':
                anchors = self.module_list[module_index][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])
                # Get the number of classes
                num_classes = int(module["classes"])
                # Transform
                x = x.data
                if CUDA:
                    x = x.cuda()
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA = torch.cuda.is_available())
                if not write_output:  # this is just to store the first scale prediction tensor to the "detections" variable
                    detections = x
                    write_output = 1

                else: # Here we already have the first scale tensor, then we just add up or "concat" the other scales
                    detections = torch.cat((detections, x), 1)

            outputs[module_index] = x

        return detections

    def load_weights(self, weight_file):
        # Open the weights file
        fp = open(weight_file, "rb")
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        # To be honest no idea regarding the next three lines
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # careful to deal with the type of the weights !!!!!!
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0

        # iterate through the network layers
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0] # assign e.g., "Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)"
                if batch_normalize:
                    bn = model[1] # assign e,g., "BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)"
                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of the model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                # we are loading weights as common because when batch normalization is present there is no bias for conv layer
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # Note: we dont have bias for conv when batch normalization is there