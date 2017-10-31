# imported module
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization 
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


# global parameter
row_size = 48
column_size = 48
channel = 1

class_num = 7
print_opt = True

# training options
train_by_cnn = False
write_model = True 

# training parameter
training_num = 1000
batch_size = 100
epoch_num = 100
learn_rate = 10

# function code
def load_image(file_dir, data_num = 'all', train_by_cnn = True):
	if data_num == 'all':
		train_file = pd.read_csv(file_dir)
		data_num = train_file.shape[0]
	else:
		train_file = pd.read_csv(file_dir, nrows = data_num)

	y_train = train_file['label'].as_matrix()
	x_train_list = train_file['feature'].tolist()

	# split data to np array
	class_statistic = np.zeros((class_num,))
	for data_idx in range(data_num):
		train_data = x_train_list[data_idx]
		data = np.array(train_data.split(), dtype = np.float32)/255
		x_train_list[data_idx] = data
		class_statistic[y_train[data_idx]]+=1

	# reshape data
	if train_by_cnn:
		x_train_shape = (data_num,row_size,column_size,channel)
	else:
		x_train_shape = (data_num,row_size*column_size*channel)
	x_train = data_reshape(x_train_list, x_train_shape)
		
	# convert y_train to categorical
	y_train = to_categorical(y_train)
	# print log 
	if print_opt:
		print('Training with %d images...\n' %y_train.shape[0])
		print('Class 0 (Angry) : %d \n' % class_statistic[0])
		print('Class 1 (Disgust) : %d \n' % class_statistic[1])
		print('Class 2 (Fear) : %d \n' % class_statistic[2])
		print('Class 3 (Happy) : %d \n' % class_statistic[3])
		print('Class 4 (Sad) : %d \n' % class_statistic[4])
		print('Class 5 (Surprise) : %d \n' % class_statistic[5])
		print('Class 6 (Neutral) : %d \n' % class_statistic[6])
	return x_train, y_train


def data_reshape(x_train_list,x_train_shape):
	x_train = np.empty(x_train_shape)	
	data_num = x_train_shape[0]
	input_shape = x_train_shape[1:]
	if len(x_train_shape) == 4:
		for idx in range(data_num):
			x_train[idx,:,:,:] = x_train_list[idx].reshape(input_shape)
	elif len(x_train_shape) == 2:
		for idx in range(data_num):
			x_train[idx,:] = x_train_list[idx]
	else:
		print('Invalid input to train deep learing model !\n')
	return x_train

def sample_weight(y_train):
	y_train = np.argmax(y_train,axis = 1)
	class_statistic = np.zeros((class_num,))
	for i in range(y_train.shape[0]):
		class_statistic[y_train[i]]+=1
	class_weight = 1/class_statistic
	class_weight = class_weight/np.sum(class_weight)
	y_train_weight = np.empty(y_train.shape)
	for i in range(y_train.shape[0]):
		y_train_weight[i] = class_weight[y_train[i]]
	return y_train_weight


def validation_split(x_train,y_train,validate_ratio = 0.3):
	rand_idx = np.random.permutation(x_train.shape[0])
	val_data_num = int(x_train.shape[0]*validate_ratio)
	x_val = x_train[rand_idx[:val_data_num]]
	y_val = y_train[rand_idx[:val_data_num]]
	x_train = x_train[rand_idx[val_data_num:]]
	y_train =  y_train[rand_idx[val_data_num:]]
	return (x_train,y_train), (x_val,y_val)

def build_cnn(input_shape):
	cnn_model = Sequential()
	cnn_model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = input_shape))
	model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(16, (3,3), activation = 'relu', padding = 'same'))
	model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(8, (1,1), activation = 'relu', padding = 'same'))
	model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	# cnn_model.add(Conv2D(8, (1,1), activation = 'relu', padding = 'same'))
	# cnn_model.add(MaxPool2D(pool_size = (2,2))) 
	cnn_model.add(Flatten())
	cnn_model.add(Dense(32, activation = 'relu'))
	cnn_model.add(Dense(32, activation = 'relu'))
	cnn_model.add(Dense(class_num, activation = 'softmax'))
	return cnn_model

def build_dnn(input_shape):
	dnn_model = Sequential()
	dnn_model.add(Dense(64, activation = 'relu', input_shape = input_shape))
	dnn_model.add(Dense(200, activation ='relu'))
	dnn_model.add(Dense(128, activation = 'relu'))
	dnn_model.add(Dense(class_num, activation='softmax'))
	return dnn_model

def trianing_model(x_train, y_train, x_val, y_val, y_train_weight = None, train_by_cnn = True):
	# build model
	if train_by_cnn:
		model = build_cnn(input_shape = x_train.shape[1:])
	else :
		model = build_dnn(input_shape = x_train.shape[1:])

	# print model summary
	if print_opt:
		model.summary()

	# fit model
	model.compile(loss = categorical_crossentropy, optimizer = Adadelta(lr = learn_rate), metrics = ['accuracy'])
	model.fit(x_train, y_train, sample_weight = y_train_weight, batch_size = batch_size, epochs = epoch_num, validation_data = (x_val,y_val), verbose = 1)

	return model

if __name__ == '__main__':
	#  load image data
	file_dir = 'data/train.csv'
	x_train, y_train = load_image(file_dir, data_num = training_num, train_by_cnn = train_by_cnn)
	training_num = y_train.shape[0]

	# validation split
	(x_train,y_train), (x_val,y_val) = validation_split(x_train,y_train)

	# build traing model 
	y_train_weight = sample_weight(y_train)
	model = trianing_model(x_train, y_train, x_val, y_val, y_train_weight,train_by_cnn)









