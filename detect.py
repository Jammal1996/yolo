# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:49:55 2022

@author: LIPS
"""

from utilities import arg_parse
from model import Darknet
import torch 
import time 
import os.path as osp
import os 
import cv2 
from torch.autograd import Variable
from utilities import preprocess_image, write_results, load_classes, draw_boxes

# getting the current directory
curr_dir = os.getcwd()

# getting parent directories 
parent_dir_1 = os.path.dirname(curr_dir)

# Initiate the arguments parsing
args = arg_parse()

# Load the images from the images directory
images = args.images

# Assign the detected images in another folder
det_images = args.det

# Load the configuration file
config_file = args.cfgfile

# Load the weights
weights_file = args.weightsfile

# Load the batch size
batch_size = int(args.bs)

# Load the confidence score (default = 0.5)
confidence = float(args.confidence)

# Load the non-max suppression value "IoU Value" (Default = 0.4)
nms_thesh = float(args.nms_thresh)

# Check for GPU availability
CUDA = torch.cuda.is_available()

# Loading classes 
num_classes = 80
classes = load_classes("coco_names.txt")

# Create a sample model
model = Darknet(config_file)

# Apply the weights to the CNN "Darknet"
model.load_weights(weights_file)

# Get the image dimension from the configuration file
model.net_info["height"] = args.reso

# get the image dimensions from the config file
conf_inp_dim = int(model.net_info["height"])

# Check-point for image dimensions
assert conf_inp_dim % 32 == 0
assert conf_inp_dim > 32

#If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

# Reading addresses timer
read_dir = time.time()

# track the directory of each single image and return a list of image directories
try:
    imgs_list_dir = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]

except NotADirectoryError:
    imgs_list_dir = []
    imgs_list_dir.append(osp.join(osp.realpath('.'), images))

except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

# Create a folder where the output images exist
if not os.path.exists(args.det):
    os.makedirs(args.det)

# loading batch timer - *Start detecting*
load_batch = time.time()

# Read the desired images and return a list of them
loaded_imgs = [cv2.imread(x) for x in imgs_list_dir]

# Use the preprocess image function to convert the image form from numpy to tensor for CNN operations
imgs_batch = list(map(preprocess_image, loaded_imgs, [conf_inp_dim for x in range(len(imgs_list_dir))]))

# Make a list of [width, height] of the corresponding images
imgs_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]

# Convert the list of the dimensions to a 2x-tensor
imgs_dim_list = torch.FloatTensor(imgs_dim_list).repeat(1,2)

leftover = 0

if (len(imgs_dim_list) % batch_size):
    leftover = 1

# WARNING ! Don't get confused with these two variables: imgs_batch and img_batches
# imgs_batch is the list of images in the images folder
# img_batches is the batch size of the Darknet CNN
if batch_size != 1:
    num_batches = len(imgs_list_dir) // batch_size + leftover
    img_batches = [torch.cat((imgs_batch[i*batch_size : min((i + 1)*batch_size,
                        len(imgs_batch))]))  for i in range(num_batches)]

write = 0

if CUDA:
    imgs_dim_list = imgs_dim_list.cuda()

# Detector loop timer
start_det_loop = time.time()
for i, batch in enumerate(imgs_batch):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        # Pass the image to the CNN
        prediction = model(Variable(batch), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
    end = time.time()
    if type(prediction) == int:
        for img_num, image in enumerate(imgs_list_dir[i * batch_size: min((i + 1) * batch_size, len(imgs_list_dir))]):
            img_id = i * batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " None "))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # transform the attribute from index in batch to index in imgs_dim_list
    if not write:  # Store the very first output (here write will be equal to zero, so the statement is true)
        output = prediction
        write = 1

    else:
        output = torch.cat((output, prediction)) # concatenate the outputs
    for img_num, image in enumerate(imgs_list_dir[i * batch_size: min((i + 1) * batch_size, len(imgs_list_dir))]):
        img_id = i * batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
    # Make sure the whole process is complete
    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print("No detections were made")
    exit()


imgs_dim_list = torch.index_select(imgs_dim_list, 0, output[:, 0].long()) # this will return the index of the objects in each individual image

scaling_factor = torch.min(608 / imgs_dim_list, 1)[0].view(-1, 1) # return the minimum value of the image dimension (height or width) according to the desired scale

output[:, [1, 3]] -= (conf_inp_dim - scaling_factor * imgs_dim_list[:, 0].view(-1, 1)) / 2 # rescale x1,x2
output[:, [2, 4]] -= (conf_inp_dim - scaling_factor * imgs_dim_list[:, 1].view(-1, 1)) / 2 # rescale y1,y2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, imgs_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, imgs_dim_list[i, 1])

output_recast = time.time()
class_load = time.time()

draw = time.time()


list(map(lambda x: draw_boxes(x, loaded_imgs), output))

det_imgs_list_dir = [osp.join(osp.realpath('.'), det_images, img) for img in os.listdir(images)]

list(map(cv2.imwrite, det_imgs_list_dir, loaded_imgs))

end = time.time() # End Detecting

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}s".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}s".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}s".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}s".format("Detection (" + str(len(imgs_list_dir)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}s".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}s".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}s".format("Average time_per_img", (end - load_batch)/len(imgs_list_dir)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()