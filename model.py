# The script used to create and train the model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# load train and test data
csv_path='../data/augmented-data/train_log.csv'
table = pd.read_csv(csv_path)
steer_train = np.array(table['steer'])
path_train = np.array(table['path'])
path_train, steer_train = shuffle(path_train, steer_train)
csv_path='../data/augmented-data/test_log.csv'
table = pd.read_csv(csv_path)
steer_test = np.array(table['steer'])
path_test = np.array(table['path'])

def generator(path_all, steer_all, batch_size=64):
    num_examples = len(path_all)
    offset = 0
    while True:
        x_batch=[]
        idx = np.mod(np.array(range(offset,offset+batch_size)),num_examples)
        for path in path_all[idx]:
            input_file_path = '../data/augmented-data/'+path
            img = plt.imread(input_file_path)
            x_batch.append((img-128)/128)
        y_batch = steer_all[idx]
        yield (np.array(x_batch),y_batch)
        offset = (offset + batch_size)%num_examples

gen_train = generator(path_train,steer_train)

#while True:
#    _,y_batch = gen_train.__next__()
#x_batch1,y_batch1 = gen_train.__next__()
#
#x_batch2,y_batch2 = gen_train.__next__()



# Model architecture definition
model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(80, 320, 3), border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(48, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Convolution2D(64, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('elu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.summary()

# Compile and train the model
model.compile(optimizer=Adam(lr=1e-4),loss='mse')
hist = model.fit_generator(gen_train, samples_per_epoch=len(steer_train), nb_epoch=3)
print(hist.history)
model.save('model_nb_epoch_3_no_steer_correction.h5')

gen_test = generator(path_test,steer_test)
out = model.evaluate_generator(gen_test,len(steer_test))
print('MSE loss = {}'.format(out))
