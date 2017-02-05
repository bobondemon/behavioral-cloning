# The script used to create and train the model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import preprocessing_utils as pp
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

# load train data
csv_path='../data/supported-data/train_log.csv'
table = pd.read_csv(csv_path)
steer = table['steering']
c_path = table['center']
l_path = table['left']
r_path = table['right']
# shuffle the train data
permute_idx = np.random.permutation(len(steer))
c_path = c_path[permute_idx]
l_path = l_path[permute_idx]
r_path = r_path[permute_idx]
steer = steer[permute_idx]

# define applying artificial effects
def apply_augmentation(img_batch,pos_batch,st_batch):
    # those probabilities of artificial effects
    use_flip = 0.5
    x_batch=[]
    y_batch=[]
    for img,pos,st in zip(img_batch,pos_batch,st_batch):
        # first do the steering angle correction according to position of camera
        st=pp.get_lr_steer_angle(st,pos)
        if np.random.uniform()<use_flip:
            img,st=pp.flip_img(img,st)
        img=pp.brighten_img(img)
        img,st=pp.hshift_img(img,st)
        #normalize and crop
        img=pp.normalize_img(pp.crop_img(img))
        x_batch.append(img)
        y_batch.append(st)
    return np.array(x_batch),np.array(y_batch)

# define the generator
def generator(c_path,l_path,r_path,steer, batch_size=64):
    num_examples = len(c_path)
    offset = 0
    while True:
        x_batch=[]
        pos_batch=[]
        idx = np.mod(np.array(range(offset,offset+batch_size)),num_examples)
        for i in idx:
            # position selection
            pos = int(np.random.uniform(0,3))
            if pos==0:
                pos='center'
                input_file_path = '../data/supported-data/'+c_path[i]
            elif pos==1:
                pos='left'
                input_file_path = '../data/supported-data/'+l_path[i]
            else:
                pos='right'
                input_file_path = '../data/supported-data/'+r_path[i]
            img = plt.imread(input_file_path)
            x_batch.append(img)
            pos_batch.append(pos)
        x_batch = np.array(x_batch)
        pos_batch = np.array(pos_batch)
        y_batch = steer[idx]
#        print('x_batch.shape={}'.format(x_batch.shape))
        x_batch, y_batch = apply_augmentation(x_batch,pos_batch,y_batch)
#        print('--x_batch.shape={}'.format(x_batch.shape))
        yield (x_batch,y_batch)
        offset = (offset + batch_size)%num_examples

batch_size=64
#gen_train = generator(c_path,l_path,r_path,steer,batch_size=batch_size)

#while True:
#    _,y_batch = gen_train.__next__()
#    print('zero count={}'.format(len(np.where(y_batch==0)[0])))
#x_batch1,y_batch1 = gen_train.__next__()
#
#x_batch2,y_batch2 = gen_train.__next__()

# Model architecture definition
model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3), subsample=(2, 2), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))

model.summary()
# Compile and train the model
model.compile(optimizer=Adam(lr=1e-4),loss='mse')
    
samples = np.ceil(len(steer)/batch_size).astype('int')*batch_size
for epoch in range(50):
    print('epoch = {}'.format(epoch))
    permute_idx = np.random.permutation(len(steer))
    c_path = c_path[permute_idx]
    l_path = l_path[permute_idx]
    r_path = r_path[permute_idx]
    steer = steer[permute_idx]
    gen_train = generator(c_path,l_path,r_path,steer,batch_size=batch_size)
    hist = model.fit_generator(gen_train, samples_per_epoch=samples, nb_epoch=1)
    print(hist.history)
    if epoch%2==0:
        model.save('model_{}_balancing.h5'.format(epoch+1))


