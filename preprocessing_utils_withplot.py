# Image-augmentation Pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math

img_dir='../data/supported-data/'
csv_path='../data/supported-data/train_log.csv'
table = pd.read_csv(csv_path)
steer = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']

select_idx = 89
src_img_c = plt.imread(img_dir+c_path[select_idx])
src_img_l = plt.imread(img_dir+l_path[select_idx])
src_img_r = plt.imread(img_dir+r_path[select_idx])
src_st = steer[select_idx]
print('select idx = '+str(select_idx))

# Cropping of the Car hood and sky from the training images.
# Note: The last step of the data preprocessing
def crop_img(img):
    h_up = 40
    h_down = img.shape[0]-20
    w_up = 20
    w_down = img.shape[1]-20
    return cv2.resize(img[h_up:h_down,w_up:w_down,:],(200, 66))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(crop_img(src_img_l))
plt.title('Left')
plt.subplot(1,3,2)
plt.imshow(crop_img(src_img_c))
plt.title('center')
plt.subplot(1,3,3)
plt.imshow(crop_img(src_img_r))
plt.title('right')
plt.tight_layout()
plt.savefig('crop_img.png')

# Normalization
def normalize_img(img):
    img=img.astype(float)
    return (img-128.0)/128.0

# Image Flipping
def flip_img(img,steer):
    return (img[:,-1::-1,:], steer*-1)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(src_img_c)
plt.title('Original img, steer = {}'.format(src_st))
plt.subplot(1,2,2)
flipped_img, flipped_st = flip_img(src_img_c,src_st)
plt.imshow(flipped_img)
plt.title('Image flipping, steer = {}'.format(flipped_st))
plt.tight_layout()

# Left/Right Camera Images
# ref: https://medium.com/@chrisgundling/cnn-model-comparison-in-udacitys-driving-simulator-9261de09b45#.6k571w66p
# left right images where offset horizontally from the center camera by approximately 60 pixels.
# Based on this information, I chose a steering angle correction of +/- 0.25 units or +/- 6.25 degrees for these left/right images.
def get_lr_steer_angle(steer,lr):
    if steer==0 or lr=='center':
        return steer
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

# left
plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(src_img_l)
plt.hold(True)
st_l = get_lr_steer_angle(src_st,'left')
plt.title('{} with angle = {}'.format('left',st_l*25))
theta = (st_l*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
# center
plt.subplot(3,1,2)
plt.imshow(src_img_c)
plt.hold(True)
st_c = get_lr_steer_angle(src_st,'center')
plt.title('{} with angle = {}'.format('center',st_c*25))
theta = (st_c*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
# right
plt.subplot(3,1,3)
plt.imshow(src_img_r)
plt.hold(True)
st_r = get_lr_steer_angle(src_st,'right')
plt.title('{} with angle = {}'.format('right',st_r*25))
theta = (st_r*25/360)*2*math.pi
plt.plot([160,np.clip(160+80*math.tan(theta),a_min=0,a_max=319)],[160,80],'r-')
plt.xlim([0,320])
plt.tight_layout()
plt.savefig('l_c_r.png')
    
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
    print('tr_x={}'.format(tr_x))
    steer=steer+0.004*tr_x
    return img,steer

plt.figure()
plt.subplot(2,1,1)
plt.imshow(crop_img(src_img_c))
plt.title('Original img, steer={}'.format(src_st))
plt.subplot(2,1,2)
hshift_img,hshift_st = hshift_img(src_img_c,src_st)
plt.imshow(crop_img(hshift_img))
plt.title('Horizontal shift img, steer={}'.format(hshift_st))
plt.tight_layout()
plt.savefig('test_hshift.png')

# Image Brightness
def brighten_img(img,low_ratio=0.5,up_ratio=1.1):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    ratio = random.uniform(low_ratio,up_ratio)
    hsv[:,:,-1] = np.clip(hsv[:,:,-1]*ratio,a_min=0,a_max=255)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(src_img_c)
plt.title('Original img')
plt.subplot(2,1,2)
brightened_img = brighten_img(src_img_c)
plt.imshow(brightened_img)
plt.title('Image brightness')
plt.tight_layout()
plt.savefig('test_brightness.png')

# Image Blurring
def blur_img(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(src_img_c)
plt.title('Original img')
plt.subplot(2,1,2)
blurred_img = blur_img(src_img_c)
plt.imshow(blurred_img)
plt.title('Image blurring')
plt.tight_layout()
plt.savefig('test_blurring.png')

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
#    
#plt.figure()
#plt.subplot(2,1,1)
#plt.imshow(c_img)
#plt.title('Original img')
#plt.subplot(2,1,2)
#rotated_img,_ = rotate_img(c_img,steer)
#plt.imshow(rotated_img)
#plt.title('Image rotation')
#plt.imsave('test_rotation',rotated_img)
#
