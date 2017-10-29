# imported module
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


# global parameter
row_size = 48
column_size = 48
channel = 1
data_num = 10
class_num = 7
input_shape = (row_size,column_size,channel)

# training parameter
batch_size = 128
epoch_num = 10

# function code
def load_image(file_dir):
	train_file = pd.read_csv(file_dir, nrows = 10)

	y_train = train_file['label'].as_matrix()
	y_train = to_categorical(y_train, class_num)
	x_train_list = train_file['feature']
	x_train = np.empty((data_num,row_size,column_size,channel))
	data_idx = 0
	for train_data in x_train_list:
		data = np.array(train_data.split(), dtype = np.float32)/255
		x_train[data_idx,:,:,:] = data.reshape(input_shape)
		data_idx += 1
	return x_train,y_train

def build_cnn():
	cnn_model = Sequential()
	cnn_model.add(Conv2D(32, (7,7), activation = 'relu', padding = 'same', input_shape = input_shape))
	# cnn_model.add(MaxPool2D(pool_size = (2,2)))
	# cnn_model.add(Drop)
	cnn_model.add(Conv2D(16, (5,5), activation = 'relu', padding = 'same'))
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(8, (3,3), activation = 'relu', padding = 'same'))
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Flatten())
	cnn_model.add(Dense(128, activation = 'relu'))
	cnn_model.add(Dense(class_num, activation = 'softmax'))
	cnn_model.summary()

	return cnn_model





if __name__ == '__main__':
	#  load image data
	file_dir = 'data/train.csv'
	x_train, y_train = load_image(file_dir)

	# build traing model 
	cnn_model = build_cnn()

	# fit model
	cnn_model.compile(loss = categorical_crossentropy, optimizer = Adadelta(), metrics = ['accuracy'])
	cnn_model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch_num ,verbose = 2)







