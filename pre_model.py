# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:59:57 2022

@author: LIPS
"""

import os 
import torch.nn as nn

# getting the current directory
curr_dir = os.getcwd()

# getting parent directories 
parent_dir_1 = os.path.dirname(curr_dir)

# the folder to get the config file
config_file = os.path.join(curr_dir, "cfg\yolov3.cfg")

class dummy_layer(nn.Module):
    def __init__(self):
        super(dummy_layer, self).__init__()

class detection_layer(nn.Module):
    def __init__(self, anchors):
        super(detection_layer, self).__init__()
        self.anchors = anchors

# The following function takes a configuration file, and returns a list of blocks. Each block describes a block in the neural
# network to be built. Block is represented as a dictionary in the list

# We need to save the content of the configuration file in a list of stings
def parse_cfg(config_file):

    """
    Takes a configuration file
    
    Returns a list of blocks. Each block describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    
    # List Preprocessing
    file = open(config_file, 'r')
    lines = file.read().split("\n")  # read the file store the lines in a list called "lines", return a list of each line in the config file as a string
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']  # get rid of the empty lines and comments
    lines = [line.lstrip().rstrip() for line in lines]  # get rid of the left and right whitespaces (didn't do a difference, but it's okay !!!)
    
    # Instantiate an empty dict for storing an individual , and an empty list
    block = {} # a full string element starting with the type of the NN operation (e.g, net, convolutional, shortcut, etc...)
    blocks = [] # a list storing all of the single operation strings
    
    # Loop over the resultant list to get blocks
    for line in lines:
        
        if line[0] == '[': # The start of a new block (i.e. [net], [convolutional], [shortcut])
            # this condition is executed to make sure that we don't store the same key value over and over again (or overwriting the variable) !!
            if len(block) != 0:  # The block is not empty, and we start a new string {'type': 'net', 'batch': '64', 'subdivisions': '16', 'width': '608', 'height': '608', 'channels': '3',.........
                blocks.append(block)
                block = {}  # emptying the dict
            block["type"] = line[1:-1].rstrip() # [1:-1] means to start from index 1 to the before last index, this is done to remove the brackets not more !

        else:
            key, value = line.split('=') # Split the single string element in two parts: key and a value (e.g. 'batch' = '64')
            block[key.rstrip()] = value.lstrip()  # removing spaces on left and right side

    blocks.append(block) # this is tricky!!!! because we need also to store the last string inside the "blocks" list: {'type': 'yolo', 'mask': '0,1,2', 'anchors': '10,13,  16,30,  33,23,  30,61,.........
    return blocks

def model_initialization(blocks):
    darknet_details = blocks[0] # Captures the input string "net" information (e.g. image height and width, learning rate, etc...)
    track_filters = 3 # starting with the number of color channels (e.g. RGB), and keep track of the number of filters in the layer on which the convolutional layer is being applied
    output_filters = []  # list of filter numbers in each layer. It is useful while defining number of filters in routing layer
    module_list = nn.ModuleList()
    for type_index, block in enumerate(blocks[1:]): # Here we have multiple strings "types" and each single string contains
        model = nn.Sequential() # Instantiate a CNN

        # In case of convolutional layer
        if (block["type"] == "convolutional"): # for example there are: [net], [convolutional], [yolo], etc....
            activation = block["activation"] # replace the activation "value" string with assigning to a new activation variable
            filters = int(block["filters"]) # replace the filters "value" string with assigning to a new filters integer variable
            kernel_size = int(block["size"]) # replace the kernel size (feature map) "value" string with assigning to a new kernel size variable
            stride = int(block["stride"]) # replace the stride "value" string with assigning to a new stride variable
            use_bias = False if ("batch_normalize" in block) else True # Check for bias term, then apply it
            pad = (kernel_size - 1) // 2 # Check for padding, otherwise return 0, and discard the reminder

            # Add a convolutional layer
            conv = nn.Conv2d(in_channels=track_filters, out_channels=filters, kernel_size=kernel_size,
                             stride=stride, padding=pad, bias=use_bias) # e.g., self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
            model.add_module(f"conv_{type_index}", conv) # e.g., network3.add_module('hidden', nn.Linear(in_features, out_features))

            # Add a batch norm layer
            if "batch_normalize" in block:
                b_norm = nn.BatchNorm2d(filters)
                model.add_module(f"batch_norm_{type_index}", b_norm)

            # Check activation function (Linear for for convolutional layers, and Leaky ReLU for YOLO)
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                model.add_module(f"leaky_{type_index}", activn)

        # In case of upsampling layer
        elif (block["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners= True)
            model.add_module(f"upsample_{type_index}", upsample)

        # In case of route layer
        elif (block["type"] == 'route'):
            block['layers'] = block['layers'].split(',') # In general we may have one or two values "range" (e.g, -4, -1)
            block['layers'][0] = int(block['layers'][0]) # Convert the string into an integer
            start = block['layers'][0] # store the first negative number in a variable
            # In case of having only one route layer value
            if len(block['layers']) == 1:
                block['layers'][0] = int(type_index + start) # add the route layer "negative number" to the current layer "type_index"
                filters = output_filters[block['layers'][0]]  # this is tricky ! here we assign the "already" or "previous" stored value in the output_filters in the current layer filters
            elif len(block['layers']) > 1:
                # e.g., we have -1,28 and present layer is 20 we have sum filters in 19th and 28th layer
                block['layers'][0] = int(type_index + start) # store the first negative number in a variable
                block['layers'][1] = (int(block['layers'][1]) - type_index) # we subtract because we want to get the difference between the current layer filters and the desired number of filters
                block['layers'][1] = int(block['layers'][1]) # verify that the variable is an integer
                filters = output_filters[block['layers'][0]] + output_filters[block['layers'][1]] # same as in the ==1 case....

            # that means this layer doesn't have any forward operation
            route = dummy_layer()
            model.add_module(f"route_{type_index}", route)

        # In case of shortcut layer
        elif block["type"] == "shortcut":
            shortcut = dummy_layer()
            model.add_module(f"shortcut_{type_index}", shortcut)

        # In case of yolo "detection" layer, for YOLO the yolo-detection comes before the route layer, very important !!!!!!
        elif block["type"] == "yolo":
            mask = block["mask"].split(",") # split and store the mask values in a list "mask"
            mask = [int(m) for m in mask] # convert the individual strings "mask values" to integers
            anchors = block["anchors"].split(",") # do exactly the same we did in the mask section above
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)] # of course we have two values (height and width) for each corresponding masks
            anchors = [anchors[i] for i in mask]

            detector_layer = detection_layer(anchors)
            model.add_module(f"Detection_{type_index}", detector_layer)

        module_list.append(model) # We add the individual model based on the if-condition
        output_filters.append(filters) # We add the filters to keep the track
        track_filters = filters # Here we need this variable to replace the first 3 RGB filters and track the filters along the network
    return darknet_details, module_list
