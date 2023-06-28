# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:40:37 2022

@author: LIPS
"""

import torch 
import numpy as np 
import cv2 
import random 
import pickle as pkl
import argparse

# =============================================================================================================================================== #
#                                                                      Parsing Arguments                                                          #
# =============================================================================================================================================== #

def arg_parse_images():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Images Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Configuration file",
                        default="yolov3.config", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="608", type=str)

    return parser.parse_args()

def arg_parse_video():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "input_video.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()

# =============================================================================================================================================== #
#                                                                        Useful Functions                                                         #
# =============================================================================================================================================== #

############## unique tensor ################
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

############## Loading YOLO "CoCo" Classes ######################

def load_classes(class_file):
    fp = open(class_file, "r")
    names = fp.read().split("\n")[:-1]
    return names

num_classes = 80
classes = load_classes("coco_names.txt")

############ Transform image tensor form 4D --> 3D ###############

def predict_transform(x, inp_dim, anchors, num_classes, CUDA):
    # The main purpose of this function is to transform the shape of the input tensor "image" from
    # 4D t-shape ---> [number of images , the bounding box coordinates + the confidence score + number of classes , image height, image width]
    # 3D t-shape ---> [number of images, number of anchors * height * width, the bounding box coordinates + the confidence score + number of classes]
    # and to transform the output of the network coordinates, height, and width to the desired output of the coordinates according to the published paper "YOLOv3: An Incremental Improvement" ----> https://arxiv.org/abs/1804.02767
    batch_size = x.size(0)
    grid_size = x.size(2)
    stride = inp_dim // x.size(2)  # stride (the factor) by which current feature map reduced from input ----> 32, 16, 8
                                   # 32 -> small scale, 16 -> medium scale, 8 -> large scale
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # [1, 255, 13, 13] ------> 80 represents the number of COCO classes,
    # 4 represents the bounding box coordinates, 1 represents the confidence score -----> the total will be 85,
    # but still we have 3 bounding box for each cell -----, then 3 * 85 will be 255 <important>
    prediction = x.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # [number of images , the bounding box coordinates + the confidence score + number of classes , image height * width]
    prediction = prediction.transpose(1, 2).contiguous()  # [number of images , the bounding box coordinates + the confidence score + number of classes , image height * width]         # contiguous used to create a new tensor and copy the same tensor (with the same order since transpose method only changes the way how the matrix tensor is organized) with new assigning
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)  # [number of images, number of anchors * height * width, the bounding box coordinates + the confidence score + number of classes]

    # the dimension of anchors is wrt original image. We will make it corresponding to the current feature map
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # Transform the x, y-bb centers and the confidence score
    # Here we sigmoid the predictions of the x-center, y-center, and the confidence score
    # **important
    # prediction[:, :, 0] all of the network x-centers of the bounding boxes
    # prediction[:, :, 1] all of the network y-centers of the bounding boxes
    # prediction[:, :, 2] all of the network width of the bounding boxes
    # prediction[:, :, 3] all of the network height of the bounding boxes
    # prediction[:, :, 4] all of the network confidence score of the bounding boxes
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size) # assign a list of top-left coordinates of the image grids
    x, y = np.meshgrid(grid, grid) # to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing
    x_offset = torch.FloatTensor(x).view(-1, 1)  # exchange the two axes
    y_offset = torch.FloatTensor(y).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1) # concat along the second axis, so we have a pair of x-y coordinates
    x_y_offset = x_y_offset.repeat(1, num_anchors) # create another 3 pairs of x-y coordinates, since we have 3 anchor boxes
    x_y_offset = x_y_offset.view(-1,2) # change the shape of the tensor to be in a way that we have all the pairs are sequentially sorted in two columns
    x_y_offset = x_y_offset.unsqueeze(0) # Add the axis of the number of images
    prediction[:, :, :2] += x_y_offset # Add the offset coordinates to the predicted x-y coordinates

    # Transform the height, and width according to the published paper
    anchors = torch.FloatTensor(anchors) # this is needed to meet the shape of the input image tensor
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1) # create another tensor but repeat the number of rows 13 * 13 times,
                                                       # and the columns only one time
    anchors = anchors.unsqueeze(0) # Add the axis of the number of images
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # width and height
    # Here we also sigmoid the predictions of the classes
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    # Finally we multiply the predicted x-y centers, width, and height with the stride to get the desired scale (small, med, or large)
    prediction[:, :, :4] *= stride

    return prediction

################## Image Resizing #####################

def canvas_image(img, conf_inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = conf_inp_dim  # dimension from configuration file
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # we fill the extra pixels with 128
    canvas = np.full((conf_inp_dim[1], conf_inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas

################### converting images from opencv format to torch format ########################

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def preprocess_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns processed image, original image and original image dimension
    """
    CUDA = torch.cuda.is_available()
    img = (canvas_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1]
    img_ = img.transpose((2, 0, 1)).copy()
    img_ = torch.as_tensor(img_).float().div(255.0).unsqueeze(0)
    if CUDA:
        img_ = img_.cuda()
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

############## Drawing Bounded Boxes ###############

colors = pkl.load(open("pallete", "rb"))

def draw_boxes(x, results):
    
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def write_video(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

# =============================================================================================================================================== #
#                                              Non - Max Suppression (Intersection over Union)                                                    #
# =============================================================================================================================================== #

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    intersection_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersection_area / (b1_area + b2_area - intersection_area)
    return iou

# =============================================================================================================================================== #
#                                                             Objectness Score                                                                    #
# =============================================================================================================================================== #

def write_results_images(prediction, confidence, num_classes, nms_conf):
    # This function filters the predicted bounding boxes by calculating the IOU with respect to each individual image
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2) # consider the confidence score predictions, which are higher than the set threshold
                                                                        # and create a third axis
                                                                        # [1, 22743, 1]
    
    prediction = prediction * conf_mask # multiply the considered confidence score predictions with the all of the predictions
    # Here we we transform the (center x, center y, width, height) attributes of our boxes, to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    # to make it easier to perform the IOU over the anchor boxes
    box_corner = prediction.new(prediction.shape) # make a new tensor with the same shape as the prediction shape or size
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    # iterate through each single image
    for ind in range(batch_size):
        image_pred = prediction[ind]    # [number of anchors * height * width, the bounding box coordinates + the confidence score + number of classes]
                                        # image Tensor, e.g, 0-index mean image #1, 1-index means image #2.............
                                        # [22743, 85]
        # confidence thresholding
        # NMS
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], 1) # get the max value along
        max_conf = max_conf.float().unsqueeze(1) # [22743, 1]
        max_conf_index = max_conf_index.float().unsqueeze(1) # [22743, 1]
        seq = (image_pred[:, :5], max_conf, max_conf_index) # gather the three tensors
        image_pred = torch.cat(seq, 1) # [22743, 7]
        non_zero_ind = (torch.nonzero(image_pred[:, 4])) # [non-zero, 1] [5,1]
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7) # view(-1,7) to make sure that we have the shape of [non-zero, 5 indices + one index (max_conf) + one index (max_conf_index)]
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
        #

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index "the last index"
        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1) # [non-zero, 7] [5,7]
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze() # [5] indices
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7) # [5,7]
            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1] # return the indices of the descending sorting
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        # Detections with 8 attributes, namely, index of the image in the batch to which the detection belongs to,
        # 4 corner coordinates, objectness score, the score of class with maximum confidence, and the index of that class.
        return output
    except:
        return 0

def write_results_video(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output
    
    