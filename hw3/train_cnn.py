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

class_num = 7
# input_shape = (row_size,column_size,channel)
print_opt = True

# training options
train_by_cnn = True
write_model = True 

# training parameter
training_num = 15000
batch_size = 100
epoch_num = 10
learn_rate = 1

# function code
def load_image(file_dir, data_num = 'all', print_log = False):
	if data_num == 'all':
		train_file = pd.read_csv(file_dir)
		data_num = train_file.shape[0]
	else:
		train_file = pd.read_csv(file_dir, nrows = data_num)

	y_train = train_file['label'].as_matrix()
	x_train_list = train_file['feature']
	class_statistic = np.zeros((class_num,))
	x_train = np.empty((data_num,row_size,column_size,channel))
	for data_idx in range(data_num):
		train_data = x_train_list[data_idx]
		data = np.array(train_data.split(), dtype = np.float32)/255
		x_train[data_idx,:,:,:] = data.reshape(input_shape)
		class_statistic[y_train[data_idx]]+=1
	
	# print log 
	if print_log:
		print('Loading data file....\n')
		print('Training with %d images...\n' %x_train.shape[0])
		print('Class 0 (Angry) : %d \n' % class_statistic[0])
		print('Class 1 (Disgust) : %d \n' % class_statistic[1])
		print('Class 2 (Fear) : %d \n' % class_statistic[2])
		print('Class 3 (Joy) : %d \n' % class_statistic[3])
		print('Class 4 (Sad) : %d \n' % class_statistic[4])
		print('Class 5 (Surprise) : %d \n' % class_statistic[5])
		print('Class 6 (Neutral) : %d \n' % class_statistic[6])
	return x_train,y_train

def build_cnn(input_shape):
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

def build_dnn(input_shape):
	dnn_model = Sequential()
	dnn_model.add(Dense(64, activation = 'relu', input_shape = input_shape))
	dnn_model.add(Dense(200, activation ='relu'))
	dnn_model.add(Dense(128, activation = 'relu'))
	dnn_model.add(Dense(class_num, activation='softmax'))
	dnn_model.summary()
	return dnn_model

if __name__ == '__main__':
	#  load image data
	file_dir = 'data/train.csv'
	x_train, y_train = load_image(file_dir, data_num = training_num, print_log = print_opt)
	y_train = to_categorical(y_train, class_num)

	# build traing model 
	if train_by_cnn:
		model = build_cnn(input_shape = x_train.shape[1:])
	else :
		x_train_column =  np.empty((data_num,row_size*column_size*channel))
		for i in range(x_train.shape[0]):
			x_train_column[i,:] = x_train[i,:,:,:].flatten()
		x_train = x_train_column
		del x_train_column
		model = build_dnn(input_shape = x_train.shape[1:])

	# fit model
	model.compile(loss = categorical_crossentropy, optimizer = Adadelta(lr = learn_rate), metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch_num ,verbose = 1)







