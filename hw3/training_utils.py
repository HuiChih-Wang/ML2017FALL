import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from training_parameter import *
from keras.utils import to_categorical


def load_image():
	train_file = pd.read_csv(train_data_path, nrows = training_num)

	y_train = train_file['label'].as_matrix()
	x_train_list = train_file['feature'].tolist()

	# split data to np array
	class_statistic = np.zeros((class_num,))
	for data_idx in range(training_num):
		train_data = x_train_list[data_idx]
		data = np.array(train_data.split(), dtype = np.float32)/255
		x_train_list[data_idx] = data
		class_statistic[y_train[data_idx]]+=1

	# reshape data
	if train_opt == 'cnn':
		x_train_shape = (training_num,row_size,column_size,channel)
	elif train_opt == 'dnn':
		x_train_shape = (training_num,row_size*column_size*channel)
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
	input_shape = x_train_shape[1:]
	if len(x_train_shape) == 4:
		for idx in range(training_num):
			x_train[idx,:,:,:] = x_train_list[idx].reshape(input_shape)
	elif len(x_train_shape) == 2:
		for idx in range(training_num):
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


def validation_split(x_train,y_train):
	val_idx_file = model_path+'ValSplitIdx'
	if model_load:
		rand_idx = np.load(val_idx_file)
	else:
		rand_idx = np.random.permutation(training_num)
		np.save(val_idx_file,rand_idx)

	val_data_num = int(training_num*validation_ratio)
	x_val = x_train[rand_idx[:val_data_num]]
	y_val = y_train[rand_idx[:val_data_num]]
	x_train = x_train[rand_idx[val_data_num:]]
	y_train =  y_train[rand_idx[val_data_num:]]
	return (x_train,y_train), (x_val,y_val)


def get_accuracy(y_true,y_predict):
	return np.mean(y_true==y_predict)

def get_confusion_matrix(y_true, y_predict):
	confusion_mat = np.zeros((class_num, class_num))
	for i in range(y_true.shape[0]):
		confusion_mat[y_true[i],y_predict[i]]+=1
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

def write_simulation_text():
	simulate_text_file = model_path + 'simulate_text.txt'
	with open(simulate_text,'w') as file:
		file.write("Extract data number: %d\n" %training_num)
		file.write("Validation ratio : %f\n" %validation_ratio)
		file.write("Training model : %s\n" %train_opt)
		file.write("Batch size : %d\n" %batch_size)
		file.write("Epoch number : %d\n" %epoch_num)
