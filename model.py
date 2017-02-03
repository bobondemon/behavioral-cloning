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

# load train data
csv_path='../data/supported-data/train_log.csv'
table = pd.read_csv(csv_path)
steer_train = np.array(table['steering'])
path_train = np.array(table['path'])
position_train = np.array(table['position'])
# shuffle the train data
permute_idx = np.random.permutation(len(steer_train))
path_train = path_train[permute_idx]
position_train = position_train[permute_idx]
position_train = position_train[permute_idx]
# load the test data
csv_path='../data/supported-data/test_log.csv'
table = pd.read_csv(csv_path)
steer_test = np.array(table['steering'])
path_test = np.array(table['path'])
position_test = np.array(table['position'])

# define applying artificial effects
def apply_augmentation(img_batch,pos_batch,st_batch):
    # those probabilities of artificial effects
    use_effects = 0.5
    use_flip = 0.33
    use_bright = use_flip + 0.33
    x_batch=[]
    y_batch=[]
    for img,pos,st in zip(img_batch,pos_batch,st_batch):
        dice = np.random.uniform()
        if dice>use_effects:
            dice = np.random.uniform()
            if dice<use_flip:
                img,st=pp.flip_img(img,st)
            elif dice<use_bright:
                img=pp.brighten_img(img)
            else:
                img,st=pp.flip_img(img,st)
                img=pp.brighten_img(img)
            img,st=pp.hshift_img(img,st)
        img=pp.normalize_img(pp.crop_img(img))
        x_batch.append(img)
        y_batch.append(st)
    return x_batch,y_batch

# define the generator
def generator(path_all, position_all, steer_all, batch_size=64):
    num_examples = len(path_all)
    offset = 0
    while True:
        x_batch=[]
        idx = np.mod(np.array(range(offset,offset+batch_size)),num_examples)
        for path in path_all[idx]:
            input_file_path = '../data/supported-data/'+path
            img = plt.imread(input_file_path)
            x_batch.append(img)
        y_batch = steer_all[idx]
        pos_batch = position_all[idx]
        x_batch, y_batch = apply_augmentation(x_batch,pos_batch,y_batch)
        yield (x_batch,y_batch)
        offset = (offset + batch_size)%num_examples

gen_train = generator(path_train,position_train,steer_train)

#while True:
#    _,y_batch = gen_train.__next__()
#    if any(y_batch!=0):
#        print('return')
#        break
#    else:
#        print('all 0')
#x_batch1,y_batch1 = gen_train.__next__()
#
#x_batch2,y_batch2 = gen_train.__next__()



## Model architecture definition
#model = Sequential()
#model.add(Convolution2D(24, 5, 5, input_shape=(80, 320, 3), border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
##model.add(Dropout(0.5))
#model.add(Activation('elu'))
#
#model.add(Convolution2D(36, 5, 5, border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
##model.add(Dropout(0.5))
#model.add(Activation('elu'))
#
#model.add(Convolution2D(48, 5, 5, border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
##model.add(Dropout(0.5))
#model.add(Activation('elu'))
#
#model.add(Convolution2D(64, 5, 5, border_mode='valid'))
#model.add(MaxPooling2D((2, 2)))
##model.add(Dropout(0.5))
#model.add(Activation('elu'))
#
#model.add(Flatten())
#model.add(Dense(100))
#model.add(Activation('elu'))
#model.add(Dense(50))
#model.add(Activation('elu'))
#model.add(Dense(10))
#model.add(Activation('elu'))
#model.add(Dense(1))
##model.add(Activation('softmax'))
#
#model.summary()
#
## Compile and train the model
#model.compile(optimizer=Adam(lr=1e-4),loss='mse')
#hist = model.fit_generator(gen_train, samples_per_epoch=len(steer_train), nb_epoch=3)
#print(hist.history)
#model.save('model_nb_epoch_3_no_steer_correction.h5')
#
#gen_test = generator(path_test,steer_test)
#out = model.evaluate_generator(gen_test,len(steer_test))
#print('MSE loss = {}'.format(out))
