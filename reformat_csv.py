# split to train and test csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

isPlot=True
kept_ratio_for_steer0 = 0.4
emphasize_range = [0.2, 1]
repeat_num = 4

csv_path='../data/supported-data/driving_log.csv'
out_dir='../data/supported-data/'
table = pd.read_csv(csv_path)
steer = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']

# first do some number counting
n_list = np.where(steer<0)[0]
n_num = len(n_list)
ne_list = [i for i in n_list if emphasize_range[0]<np.abs(steer[i])<emphasize_range[1]]
ne_num = len(ne_list)
p_list = np.where(steer>0)[0]
p_num = len(p_list)
pe_list = [i for i in p_list if emphasize_range[0]<np.abs(steer[i])<emphasize_range[1]]
pe_num = len(pe_list)
zero_list = np.where(steer==0)[0]
zero_num = len(zero_list)
print('Total={}, zero={}, pos={}, pos_e={}, neg={}, neg_e={}'.format(len(steer),zero_num,p_num,pe_num,n_num,ne_num))
if isPlot:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(steer,bins=21,range=(-1,1))
    plt.xlabel('Steering Angle')
    plt.ylabel('Number of Instances')
    plt.title('Before Selection')

# gen balanced dataset
f_train = open(out_dir+'train_log.csv', 'w')
f_train.write('center,left,right,steering\n')
for i in range(len(steer)):
    if steer[i]==0 and np.random.uniform()<kept_ratio_for_steer0:  # will left only kept_ratio_for_steer0 portion of data with steer 0
        f_train.write('{},{},{},{}\n'.format(c_path[i].strip(),l_path[i].strip(),r_path[i].strip(),steer[i]))
    if steer[i]!=0:
        f_train.write('{},{},{},{}\n'.format(c_path[i].strip(),l_path[i].strip(),r_path[i].strip(),steer[i]))
        if emphasize_range[0]<np.abs(steer[i])<emphasize_range[1]:
            for k in range(repeat_num):
                f_train.write('{},{},{},{}\n'.format(c_path[i].strip(),l_path[i].strip(),r_path[i].strip(),steer[i]))
    
f_train.close()

# then number counting agian
table = pd.read_csv(out_dir+'train_log.csv')
steer = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']
n_list = np.where(steer<0)[0]
n_num = len(n_list)
ne_list = [i for i in n_list if emphasize_range[0]<np.abs(steer[i])<emphasize_range[1]]
ne_num = len(ne_list)
p_list = np.where(steer>0)[0]
p_num = len(p_list)
pe_list = [i for i in p_list if emphasize_range[0]<np.abs(steer[i])<emphasize_range[1]]
pe_num = len(pe_list)
zero_list = np.where(steer==0)[0]
zero_num = len(zero_list)
print('Total={}, zero={}, pos={}, pos_e={}, neg={}, neg_e={}'.format(len(steer),zero_num,p_num,pe_num,n_num,ne_num))
if isPlot:
    plt.subplot(1,2,2)
    plt.hist(steer,bins=21,range=(-1,1),color='orange')
    plt.ylim(0,5000)
    plt.xlabel('Steering Angle')
    plt.ylabel('Number of Instances')
    plt.title('After Selection')