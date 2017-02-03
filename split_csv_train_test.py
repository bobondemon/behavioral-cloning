# split to train and test csv
import pandas as pd
import random

test_ratio = 0.05

csv_path='../data/supported-data/driving_log.csv'
out_dir='../data/supported-data/'
table = pd.read_csv(csv_path)
steer_all = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']

f_train = open(out_dir+'train_log.csv', 'w')
f_test = open(out_dir+'test_log.csv', 'w')
f_train.write('path,position,steering\n')
f_test.write('path,position,steering\n')
for i in range(len(steer_all)):
    steer = steer_all[i]
    # center image
    position = 'center'
    img_path = c_path[i].strip()
    # to train or to test
    if random.uniform(0,1)<test_ratio:
        f_test.write('{},{},{}\n'.format(img_path,position,steer))
    else:
        f_train.write('{},{},{}\n'.format(img_path,position,steer))
    
    # left image
    position = 'left'
    img_path = l_path[i].strip()
    f_train.write('{},{},{}\n'.format(img_path,position,steer))
    
    # right image
    position = 'right'
    img_path = r_path[i].strip()
    f_train.write('{},{},{}\n'.format(img_path,position,steer))

f_train.close()
f_test.close()