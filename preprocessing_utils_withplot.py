# Image-augmentation Pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math

img_dir='../data/supported-data/'
csv_path='../data/supported-data/driving_log.csv'
df = pd.read_csv(csv_path)
steer_all = df['steering']
c_path = df['center']
l_path = df['left']
r_path = df['right']
# do some analysis of csv
plt.figure(figsize=(10,7))
plt.bar(range(len(steer_all)),steer_all)
plt.axis('tight')
plt.ylim((-1,1))
plt.figure()

maxi = np.argmax(steer_all)
mini = np.argmin(steer_all)
select_idx = [i for i in np.where(steer_all>-0.8)[0] if steer_all[i]<-0.6][0]
randi = int(random.uniform(0,len(steer_all)))

c_img = plt.imread(img_dir+c_path[select_idx].strip())
l_img = plt.imread(img_dir+l_path[select_idx].strip())
r_img = plt.imread(img_dir+r_path[select_idx].strip())
steer = steer_all[select_idx]

# Image Flipping
def flip_img(img,steer):
    return (img[:,-1::-1,:], steer*-1)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(r_img)
plt.title('Original img')
plt.subplot(2,1,2)
flipped_img, _ = flip_img(r_img,steer)
plt.imshow(flipped_img)
plt.title('Image flipping')

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
#    return np.clip(rtn_theta/25,a_min=-1,a_max=1)
    return rtn_theta/25

plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(l_img)
plt.title('left')
plt.hold(True)
steer_l = get_lr_steer_angle(steer,'left')
plt.title('left with angle = {}'.format(steer_l*25))
theta = (steer_l*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
plt.subplot(3,1,2)
plt.imshow(c_img)
plt.title('center with angle = {}'.format(steer*25))
plt.hold(True)
theta = (steer*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
plt.subplot(3,1,3)
plt.imshow(r_img)
plt.hold(True)
steer_r = get_lr_steer_angle(steer,'right')
plt.title('right with angle = {}'.format(steer_r*25))
theta = (steer_r*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
plt.tight_layout()
plt.savefig('l_c_r.png')
    
# Horizontal/Vertical Shifting

# Image Brightness
def brighten_img(img,low_ratio=0.65,up_ratio=1.2):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    ratio = random.uniform(low_ratio,up_ratio)
#    print('ratio={}'.format(ratio))
    hsv[:,:,-1] = np.clip(hsv[:,:,-1]*ratio,a_min=0,a_max=255)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(c_img)
plt.title('Original img')
plt.subplot(2,1,2)
brightened_img = brighten_img(c_img)
plt.imshow(brightened_img)
plt.title('Image brightness')
plt.imsave('test_brightness',brightened_img)

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
    
plt.figure()
plt.subplot(2,1,1)
plt.imshow(c_img)
plt.title('Original img')
plt.subplot(2,1,2)
rotated_img,_ = rotate_img(c_img,steer)
plt.imshow(rotated_img)
plt.title('Image rotation')
plt.imsave('test_rotation',rotated_img)

# Cropping of the Car hood and sky from the training images.
# Note: The last step of the data preprocessing
def crop_img(img):
    h_up = 60
    h_down = img.shape[0]-20
    w_up = 0
    w_down = img.shape[1]
    return img[h_up:h_down,w_up:w_down,:]

plt.figure()
plt.subplot(1,3,1)
plt.imshow(crop_img(l_img))
plt.title('left')
plt.subplot(1,3,2)
plt.imshow(crop_img(c_img))
plt.title('center')
plt.subplot(1,3,3)
plt.imshow(crop_img(r_img))
plt.title('right')
plt.tight_layout()

# Normalization
def normalize_img(img):
    return (img-128)/128