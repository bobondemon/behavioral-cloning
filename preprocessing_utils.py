## Image-augmentation Pre-processing
import numpy as np
import cv2
import random
import math

# Image Flipping
def flip_img(img,steer):
    return (img[:,-1::-1,:], steer*-1)

# Left/Right Camera Images
# ref: https://medium.com/@chrisgundling/cnn-model-comparison-in-udacitys-driving-simulator-9261de09b45#.6k571w66p
# left right images where offset horizontally from the center camera by approximately 60 pixels.
# Based on this information, I chose a steering angle correction of +/- 0.25 units or +/- 6.25 degrees for these left/right images.

def get_lr_steer_angle(steer,lr):
    left_offset=20
    right_offset=20
    if steer==0:
        return steer
    theta = (steer*25/360)*2*math.pi
    end_point = (np.clip(160+80*math.tan(theta),a_min=0,a_max=319), 80)
    assert ((lr=='right') | (lr=='left'))
    if steer<0 and lr=='left': # turn left and with left camera
        return steer
    if steer>0 and lr=='right': # turn right and with right camera
        return steer
    if lr=='left':
        start_point = (160-left_offset,160)
    else:
        start_point = (160+right_offset,160)
    diffx = end_point[0] - start_point[0]
    diffy = start_point[1] - end_point[1]
    rtn_theta = 360*math.atan(diffx/diffy)/(2*math.pi)
    return rtn_theta/25
#    return np.clip(rtn_theta/25,a_min=-1,a_max=1)

# Horizontal/Vertical Shifting

# Image Brightness
def brighten_img(img,low_ratio=0.65,up_ratio=1.2):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    ratio = random.uniform(low_ratio,up_ratio)
    hsv[:,:,-1] = np.clip(hsv[:,:,-1]*ratio,a_min=0,a_max=255)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

# Image Blurring

# Image Rotations
# Dont use it, since the correction of angle is not done
def rotate_img(img,steer):
    height, width, channel = img.shape
    image_center = (width / 2, height / 2)
    angle = random.random()*5
    if (random.random()-0.5)<0:
        angle *= -1
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_mat, (width, height))
    return (rotated_img, steer)
    
# Cropping of the Car hood and sky from the training images.
# Note: The last step of the data preprocessing
def crop_img(img):
    h_up = 60
    h_down = img.shape[0]-20
    w_up = 0
    w_down = img.shape[1]
    return img[h_up:h_down,w_up:w_down,:]

# Normalization
def normalize_img(img):
    return (img-128)/128