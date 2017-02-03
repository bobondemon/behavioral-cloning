# producing the augmented data
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing_utils as pp
import random

img_dir='../data/supported-data/'
out_dir='../data/augmented-data/IMG/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
#save_img = False
save_img = True
emphasize_range=[0.2, 0.8]
test_ratio = 0.1

csv_path='../data/supported-data/driving_log.csv'
table = pd.read_csv(csv_path)
steer_all = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']

f_train = open(out_dir+'../train_log.csv', 'w')
f_test = open(out_dir+'../test_log.csv', 'w')
f_train.write('path,steer\n')
f_test.write('path,steer\n')
for i in range(len(steer_all)):
    is_emphasized=False
    # center image
    img_path = img_dir+c_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
    # to train or to test
    if random.uniform(0,1)<test_ratio:
        f_test.write('{},{}\n'.format('IMG/c_{}_b.jpg'.format(i),steer))
    else:
        f_train.write('{},{}\n'.format('IMG/c_{}_b.jpg'.format(i),steer))
    # emphasize image
    if emphasize_range[0]<np.abs(steer)<emphasize_range[1]:
        is_emphasized=True
        f_train.write('{},{}\n'.format('IMG/c_{}_be.jpg'.format(i),steer))
    # flip the image
    flipped_img, steer_flip = pp.flip_img(img,steer)
    f_train.write('{},{}\n'.format('IMG/c_{}_bf.jpg'.format(i),steer_flip))
    
    if save_img:
        plt.imsave(out_dir+'c_{}_b.jpg'.format(i),pp.crop_img(img))
        if is_emphasized:
            plt.imsave(out_dir+'c_{}_be.jpg'.format(i),pp.crop_img(pp.brighten_img(img)))
        plt.imsave(out_dir+'c_{}_bf.jpg'.format(i),pp.crop_img(flipped_img))
    
    # left image
    img_path = img_dir+l_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
    steer = pp.get_lr_steer_angle(steer,'left')
    f_train.write('{},{}\n'.format('IMG/l_{}_b.jpg'.format(i),steer))
    # emphasize image
    if is_emphasized:
        f_train.write('{},{}\n'.format('IMG/l_{}_be.jpg'.format(i),steer))
    flipped_img, steer_flip = pp.flip_img(img,steer)
    f_train.write('{},{}\n'.format('IMG/l_{}_bf.jpg'.format(i),steer_flip))
    
    if save_img:
        plt.imsave(out_dir+'l_{}_b.jpg'.format(i),pp.crop_img(img))
        if is_emphasized:
            plt.imsave(out_dir+'l_{}_be.jpg'.format(i),pp.crop_img(pp.brighten_img(img)))
        plt.imsave(out_dir+'l_{}_bf.jpg'.format(i),pp.crop_img(flipped_img))
    
    # right image
    img_path = img_dir+r_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
    steer = pp.get_lr_steer_angle(steer,'right')
    f_train.write('{},{}\n'.format('IMG/r_{}_b.jpg'.format(i),steer))
    # emphasize image
    if is_emphasized:
        f_train.write('{},{}\n'.format('IMG/r_{}_be.jpg'.format(i),steer))
    flipped_img, steer_flip = pp.flip_img(img,steer)
    f_train.write('{},{}\n'.format('IMG/r_{}_bf.jpg'.format(i),steer_flip))
    
    if save_img:
        plt.imsave(out_dir+'r_{}_b.jpg'.format(i),pp.crop_img(img))
        if is_emphasized:
            plt.imsave(out_dir+'r_{}_be.jpg'.format(i),pp.crop_img(pp.brighten_img(img)))
        plt.imsave(out_dir+'r_{}_bf.jpg'.format(i),pp.crop_img(flipped_img))

f_train.close()
f_test.close()