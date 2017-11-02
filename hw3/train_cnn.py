# imported module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization 
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta,Adam,SGD


# global parameter
class_list = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
row_size = 48
column_size = 48
channel = 1

class_num = 7
print_opt = True
plot_opt = False

# training options
train_by_cnn = True
write_model = True

# training parameter
activate_method = 'relu'
training_num = 28000
batch_size = 10
epoch_num = 100
opt_method = Adam(lr = 0.1)

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
		for i in range(len(class_list)):
			print('Class %d (%s) : %d \n' %(i, class_list[i], class_statistic[i]))
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
	cnn_model.add(Conv2D(32, (3,3), activation = activate_method, padding = 'same', input_shape = input_shape))
	cnn_model.add(BatchNormalization())
	# cnn_model.add(Dropout(0.25))
	#cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(64, (3,3), activation = activate_method, padding = 'same'))
	cnn_model.add(BatchNormalization())
	# cnn_model.add(Dropout(0.25))
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(128, (3,3), activation = activate_method, padding = 'same'))
	cnn_model.add(BatchNormalization())
	# cnn_model.add(Dropout(0.25))
	#cnn_model.add(MaxPool2D(pool_size = (2,2)))
	# cnn_model.add(Conv2D(8, (1,1), activation = 'relu', padding = 'same'))
	# cnn_model.add(MaxPool2D(pool_size = (2,2))) 
	cnn_model.add(Flatten())
	cnn_model.add(Dense(1000, activation = 'relu'))
	cnn_model.add(Dropout(0.25))
	cnn_model.add(Dense(1000, activation = 'relu'))
	cnn_model.add(Dropout(0.25))
	cnn_model.add(Dense(1000, activation = 'relu'))
	cnn_model.add(Dropout(0.25))
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
	model.compile(loss = categorical_crossentropy, optimizer = opt_method, metrics = ['accuracy'])
	model.fit(x_train, y_train, sample_weight = y_train_weight, batch_size = batch_size, epochs = epoch_num, validation_data = (x_val,y_val), verbose = int(print_opt))
	return model

def get_confusion_matrix(y_val_predict, y_val):
	confusion_mat = np.zeros((class_num, class_num))
	for i in range(y_val.shape[0]):
		confusion_mat[y_val[i],y_val_predict[i]]+=1
	for i in range(class_num):
		confusion_mat[i,:]/=np.sum(confusion_mat[i,:])
	return confusion_mat

def plot_confusion_mat(confusion_mat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion_mat, interpolation='nearest')
	fig.colorbar(cax)
	ax.set_xticklabels(['']+class_list)
	ax.set_yticklabels(['']+class_list)

	for (i,j),z in np.ndenumerate(confusion_mat):
		ax.text(j,i, '{:0.2f}'.format(z),ha='center',va='center')
	plt.show()


if __name__ == '__main__':
	#  load image data
	file_dir = 'data/train.csv'
	x_train, y_train = load_image(file_dir, data_num = training_num, train_by_cnn = train_by_cnn)
	training_num = y_train.shape[0]

	# validation split
	(x_train,y_train), (x_val,y_val) = validation_split(x_train,y_train)

	# build traing model 
	y_train_weight = sample_weight(y_train)
	y_train_weight = None
	model = trianing_model(x_train, y_train, x_val, y_val, y_train_weight,train_by_cnn)

	# plot confusion matrix
	if print_opt: 
		y_val_class = np.argmax(y_val, axis = 1)
		y_val_predict = model.predict_classes(x_val) 
		cm = get_confusion_matrix(y_val_class,y_val_class)
		print(cm)
		plot_confusion_mat(cm)



	# save model in h5py
	if write_model:
		model.save('traing_cnn.h5') 










