# imported module
import sys
import numpy as np
import pandas as pd
import pickle as pk
np.set_printoptions(precision = 5, suppress = True)


validate_ratio = 0.3
# function code
def read_train(train_file_name):
	x_train = pd.read_csv(train_file_name['feature'])
	x_train = x_train.as_matrix()
	y_train = pd.read_csv(train_file_name['label'])
	y_train = y_train.as_matrix().astype('float')
	# add bias term 
	x_train = np.c_[np.ones((x_train.shape[0],1)), x_train]
	return x_train, y_train

def sigmoid(value):
	return 1/(1+np.exp(-value))


def prob_predict(x_train, w):
	return sigmoid(x_train @ w)

def validate_split(x_train,y_train,validate_ratio = 0.3):
	data_num = x_train.shape[0]
	validate_num = int(validate_ratio * data_num)
	rand_seq = np.random.permutation(data_num)

	x_val = x_train[rand_seq[:validate_num],:]
	y_val = y_train[rand_seq[:validate_num],:]
	x_train = x_train[rand_seq[validate_num:],:]
	y_train = y_train[rand_seq[validate_num:],:]
	return x_train, y_train, x_val, y_val

def train_generative(x_train, y_train):
	# spilt into two class
	x_train_1 = x_train[(y_train==1).flatten(),1:]
	x_train_0 = x_train[(y_train==0).flatten(),1:]

	mean_1 = np.mean(x_train_1, axis = 0)[:,None]
	mean_0 = np.mean(x_train_0, axis = 0)[:,None]
	cov_1 = np.cov(x_train_1.T)
	cov_0 = np.cov(x_train_0.T)
	ratio_1 = np.mean(y_train)
	cov = ratio_1*cov_1 + (1-ratio_1)*cov_0
	cov_inv = np.linalg.pinv(cov)

	w = (mean_1-mean_0).T @ cov_inv
	b = -0.5 * mean_1.T @ cov_inv @ mean_1 +0.5 * mean_0.T @ cov_inv @ mean_0 + np.log(ratio_1/(1-ratio_1))
	return np.r_[b,w.T]

def categorical_predict(x_train,w):
	f = prob_predict(x_train,w)
	y_predict = np.zeros((x_train.shape[0],1))
	y_predict[f>=0.5] = 1
	return y_predict

def categorical_accuracy(y_predict_label, y_true_label):
	return np.mean(y_predict_label == y_true_label)

# main code
if __name__ == '__main__' :
	
	# read training file
	feature_file = '../data/X_train'
	label_file = '../data/Y_train'
	train_file_name = {'feature':feature_file, 'label':label_file}
	x_train, y_train = read_train(train_file_name)


	"Spilt Training and Validation Set"
	if validate_ratio is not 0:
		x_train, y_train, x_val, y_val = validate_split(x_train,y_train, validate_ratio = validate_ratio)


	"Training Generative Model"
	w_opt = train_generative(x_train,y_train)


	"Prediction Error"
	# predict training set
	y_predict_trian = categorical_predict(x_train,w_opt)
	train_accuracy = categorical_accuracy(y_predict_trian,y_train)
	print('Training Accuracy : %f' %train_accuracy)

	# predict validation set
	if validate_ratio is not 0:
		y_predict_val = categorical_predict(x_val,w_opt)
		val_accuracy = categorical_accuracy(y_predict_val,y_val)
		print('Validation Accuracy : %f' %val_accuracy)


	" Save Generative Model"
	model = {}
	model['w_opt'] = w_opt
	write_file_name = '../ouput_file/generative_modle.pickle'
	with open(write_file_name, 'wb') as write_file:
		pk.dump(model,write_file)
