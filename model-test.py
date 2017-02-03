# model testing with test set
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

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

model = load_model('model_nb_epoch_3_no_steer_correction.h5')
gen_test = generator(path_test,steer_test)
out = model.evaluate_generator(gen_test,len(steer_test))
print('MSE loss = {}'.format(out))