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
    if lr=='center':    # no correction
        return steer
        
#    if lr=='left':
#        return steer+0.25
#    else:
#        return steer-0.25
        
    offset=5
    theta = (steer*25/360)*2*math.pi
    end_point = (np.clip(160+80*math.tan(theta),a_min=0,a_max=319), 80)
#    if steer<=0 and lr=='left': # turn left and with left camera
#        offset = 5
#    if steer>=0 and lr=='right': # turn right and with right camera
#        offset = 5
    if lr=='left':
        start_point = (160-offset,160)
    else:
        start_point = (160+offset,160)
    diffx = end_point[0] - start_point[0]
    diffy = start_point[1] - end_point[1]
    rtn_theta = 360*math.atan(diffx/diffy)/(2*math.pi)
    return np.clip(rtn_theta/25,a_min=-1,a_max=1)
#    return rtn_theta/25

# Horizontal/Vertical Shifting
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.zem65mq24
# We added 0.004 steering angle units per pixel shift to the right
# and subtracted 0.004 steering angle units per pixel shift to the left.
def hshift_img(img,steer):
    max_shift_range=20
    # if tr_x > 0, the image will shift right
    tr_x = int(np.random.uniform(max_shift_range*2)-max_shift_range)
    tr_y=0
    trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols = img.shape[:2]
    img = cv2.warpAffine(img,trans_M,(cols,rows))
    steer=steer+0.004*tr_x
    return img,steer

# Image Brightness
def brighten_img(img,low_ratio=0.5,up_ratio=1.1):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    ratio = random.uniform(low_ratio,up_ratio)
    hsv[:,:,-1] = np.clip(hsv[:,:,-1]*ratio,a_min=0,a_max=255)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

# Image Blurring
def blur_img(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)
    
# This function is referenced from: https://github.com/windowsub0406/SelfDrivingCarND/blob/master/SDC_project_3/model.ipynb
def generate_shadow(image, min_alpha=0.5, max_alpha = 0.75):
    """generate random shadow in random region"""
    rows, cols, _ = image.shape
    top_x, bottom_x = np.random.randint(0, cols, 2)
    
    shadow_img = image.copy()
    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
    mask = image.copy()
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (0,) * channel_count
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    rand_alpha = np.random.uniform(min_alpha, max_alpha)
    cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)

    return shadow_img

## Image Rotations
## Dont use it, since the correction of angle is not done
#def rotate_img(img,steer):
#    height, width, channel = img.shape
#    image_center = (width / 2, height / 2)
#    angle = random.random()*5
#    if (random.random()-0.5)<0:
#        angle *= -1
#    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
#    rotated_img = cv2.warpAffine(img, rotation_mat, (width, height))
#    return (rotated_img, steer)
    
# Cropping of the Car hood and sky from the training images.
# Note: The last step of the data preprocessing
def crop_img(img):
    h_up = 40
    h_down = img.shape[0]-20
    w_up = 20
    w_down = img.shape[1]-20
    return cv2.resize(img[h_up:h_down,w_up:w_down,:],(200, 66))

# Normalization
def normalize_img(img):
    img=img.astype(float)
    return (img-128.0)/128.0