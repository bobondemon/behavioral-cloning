# producing the augmented data
import os
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing_utils as pp

img_dir='../data/supported-data/'
out_dir='../data/augmented-data/IMG/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
save_img = False
#save_img = True
csv_path='../data/supported-data/driving_log.csv'
table = pd.read_csv(csv_path)
steer_all = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']

f = open(out_dir+'../log.csv', 'w')
f.write('path,steer\n')
for i in range(len(steer_all)):
    # center image
    img_path = img_dir+c_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
    f.write('{},{}\n'.format('IMG/c_{}_b.jpg'.format(i),steer))
    flipped_img, steer = pp.flip_img(img,steer)
    f.write('{},{}\n'.format('IMG/c_{}_bf.jpg'.format(i),steer))
    
    if save_img:
        img = pp.crop_img(img)
        flipped_img = pp.crop_img(flipped_img)
        plt.imsave(out_dir+'c_{}_b.jpg'.format(i),img)
        plt.imsave(out_dir+'c_{}_bf.jpg'.format(i),flipped_img)
    
    # left image
    img_path = img_dir+l_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
#    steer = pp.get_lr_steer_angle(steer,'left')
    f.write('{},{}\n'.format('IMG/l_{}_b.jpg'.format(i),steer))
    flipped_img, steer = pp.flip_img(img,steer)
    f.write('{},{}\n'.format('IMG/l_{}_bf.jpg'.format(i),steer))
    
    if save_img:
        img = pp.crop_img(img)
        flipped_img = pp.crop_img(flipped_img)
        plt.imsave(out_dir+'l_{}_b.jpg'.format(i),img)
        plt.imsave(out_dir+'l_{}_bf.jpg'.format(i),flipped_img)
    
    # right image
    img_path = img_dir+r_path[i].strip()
    img = plt.imread(img_path)
    img = pp.brighten_img(img)
    steer = steer_all[i]
#    steer = pp.get_lr_steer_angle(steer,'right')
    f.write('{},{}\n'.format('IMG/r_{}_b.jpg'.format(i),steer))
    flipped_img, steer = pp.flip_img(img,steer)
    f.write('{},{}\n'.format('IMG/r_{}_bf.jpg'.format(i),steer))
    
    if save_img:
        img = pp.crop_img(img)
        flipped_img = pp.crop_img(flipped_img)
        plt.imsave(out_dir+'r_{}_b.jpg'.format(i),img)
        plt.imsave(out_dir+'r_{}_bf.jpg'.format(i),flipped_img)

f.close()