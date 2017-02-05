# load the training csv and rebalancing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

emphasize_range = [0.2, 1]
csv_path='../data/supported-data/train_log.csv'
out_path='../data/supported-data/train_balanced_log.csv'
table = pd.read_csv(csv_path)
steer = np.array(table['steering'])
path = np.array(table['path'])
position = np.array(table['position'])

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
plt.figure()
plt.hist(steer,bins=20,range=(-1,1))

# gen balanced dataset
f_out = open(out_path, 'w')
f_out.write('path,position,steering\n')
for i in range(len(steer)):
    if steer[i]!=0 or np.random.uniform()<0.3:  # will ignore about a half of steering 0 samples
        f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
        if i in pe_list or i in ne_list:    # emphasize the turning examples
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
        if 0.4<steer[i]:
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
            f_out.write('{},{},{}\n'.format(path[i],position[i],steer[i]))
    
f_out.close()


# plot the balanced results
table = pd.read_csv(out_path)
steer = np.array(table['steering'])
path = np.array(table['path'])
position = np.array(table['position'])

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
plt.figure()
plt.hist(steer,bins=20,range=(-1,1))