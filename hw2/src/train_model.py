# imported module
import sys
import numpy as np
import pandas as pd
import pickle as pk
np.set_printoptions(precision = 5, suppress = True)

# training parameter
feature_power = 1
validate_ratio = 0.3
regular_par = 0
iter_num = 1000
learn_rate = 1
opt_method = 'ada_grad'

# function code
def read_train(train_file_name):
	x_train = pd.read_csv(train_file_name['feature'])
	x_train = x_train.as_matrix()
	y_train = pd.read_csv(train_file_name['label'])
	y_train = y_train.as_matrix().astype('float')
	# add bias term 
	x_train = np.c_[np.ones((x_train.shape[0],1)), x_train]
	return x_train, y_train

def data_boosting(x_train,y_train):
	data_num_1 = np.sum(y_train == 1)
	data_num_0 = np.sum(y_train == 0)
	sample_num = abs(data_num_0-data_num_1)
	x_train_1 = x_train[(y_train==1).flatten(),:]
	x_train_0 = x_train[(y_train==0).flatten(),:]

	if data_num_1 > data_num_0:
		sample_idx = np.random.permutation(data_num_0)[:sample_num]
		x_train_extra = x_train_0[sample_idx,:]
		y_train_extra = np.zeros((sample_num,1))
		x_train = np.r_[x_train, x_train_extra]
		y_train = np.r_[y_train, y_train_extra]
	elif data_num_1 < data_num_0:
		sample_idx = np.random.permutation(data_num_1)[:sample_num]
		x_train_extra = x_train_1[sample_idx,:]
		y_train_extra = np.ones((sample_num,1))
		# print(x_train.shape,x_train_extra.shape)

		x_train = np.r_[x_train,x_train_extra]
		y_train = np.r_[y_train, y_train_extra]

	return x_train, y_train


def validate_split(x_train,y_train,validate_ratio = 0.3):
	data_num = x_train.shape[0]
	validate_num = int(validate_ratio * data_num)
	rand_seq = np.random.permutation(data_num)

	x_val = x_train[rand_seq[:validate_num],:]
	y_val = y_train[rand_seq[:validate_num],:]
	x_train = x_train[rand_seq[validate_num:],:]
	y_train = y_train[rand_seq[validate_num:],:]
	return x_train, y_train, x_val, y_val



def polynomial_feature(x_train, power = 3):
	x_train_poly = x_train
	for idx in range(2,power+1):
		x_train_poly = np.c_[x_train_poly, x_train[:,1:]**idx]
	return x_train_poly


def feature_normalize(x_train, train_mean, train_std):
	x_train_normalize = np.empty(x_train.shape)
	for idx in range(x_train.shape[1]): 
		if train_std[idx] != 0:
			x_train_normalize[:,[idx]] = (x_train[:,[idx]]-train_mean[idx])/train_std[idx]
		else:
			x_train_normalize[:,[idx]] = x_train[:,[idx]]
	return x_train_normalize


def sigmoid(value):
	value = np.clip(value,-14,14)
	return 1/(1+np.exp(-value))


def prob_predict(x_train, w):
	return sigmoid(x_train @ w)


def categorical_error(x_train,y_train,w):
	data_num = y_train.shape[0]
	f = prob_predict(x_train,w)
	class1_idx = y_train==1
	class0_idx = y_train==0
	eps = 1e-4
	error = -(np.sum(np.log(1-f[class0_idx]+eps)) +np.sum(np.log(f[class1_idx]+eps)))/data_num
	error += 0.5*regular_par*np.sum(w[1:,:]**2)
	return error


def categorical_gradient(x_train,y_train,w):
	data_num = x_train.shape[0]
	f = prob_predict(x_train,w)
	diff = f-y_train
	w_reg = w
	w_reg[0,:] = 0
	grad = x_train.T @ diff/data_num+regular_par*w_reg
	return grad


def grad_descend(x_train, y_train ,w_init, iter_num = 1000, learn_r =0.1, optimizer = 'normal'):
	w = w_init 
	regress_error = np.empty((iter_num,1))
	
	if optimizer is 'ada_grad':
		grad_prev = categorical_gradient(x_train,y_train,w_init)
		for iter in range(iter_num):
			# ada grad
			grad_prev_rms = np.sqrt(np.sum(grad_prev**2))
			grad = categorical_gradient(x_train, y_train, w)
			w = w - learn_r * grad/grad_prev_rms
			grad_prev = np.c_[grad_prev, grad]
			regress_error[iter,:] = categorical_error(x_train,y_train,w)
	elif optimizer is 'normal' : 
		for iter in range(iter_num):
			grad = categorical_gradient(x_train,y_train,w)
			grad_norm = np.linalg.norm(grad)
			w = w - learn_r*grad/grad_norm
			regress_error[iter,:] = categorical_error(x_train,y_train,w)
	# print(regress_error)
	return w, regress_error


def batch_gradient_descend(x_train, y_train, w_init, batch_size = 100, epoch_num = 100, learn_rate = 0.1):
	data_num = x_train.shape[0]
	batch_num = int(data_num/batch_size)
	w = w_init
	regress_error = np.empty((epoch_num,1))
	for epoch_idx in range(epoch_num):
		# shuffle data
		shuffle_idx = np.random.permutation(data_num)
		for batch_idx in range(batch_num):
			sample_start = batch_idx*batch_size
			sample_end = min(data_num,(batch_idx+1)*batch_size)
			sample_idx = shuffle_idx[sample_start:sample_end]
			x_train_batch = x_train[sample_idx,:]
			y_train_batch = y_train[sample_idx,:]
			# gradient descend
			grad = categorical_gradient(x_train_batch,y_train_batch,w)
			w = w - learn_rate * grad

		regress_error[epoch_idx,:] = categorical_error(x_train, y_train, w)
		print(regress_error[epoch_idx,:])
	return w, regress_error


def categorical_predict(x_train,w):
	f = prob_predict(x_train,w)
	y_predict = np.zeros((x_train.shape[0],1))
	y_predict[f>=0.5] = 1
	return y_predict


def categorical_accuracy(y_predict_label, y_true_label):
	return np.mean(y_predict_label == y_true_label)


# main code
if __name__=='__main__':
	# read training file
	feature_file = '../data/X_train'
	label_file = '../data/Y_train'
	train_file_name = {'feature':feature_file, 'label':label_file}
	x_train, y_train = read_train(train_file_name)

	"Data preprocessing"
	# data boosting
	x_train, y_train = data_boosting(x_train,y_train)

	# feature normalization
	x_mean = np.mean(x_train, axis = 0)
	x_std = np.std(x_train, axis = 0)
	x_train = feature_normalize(x_train,x_mean,x_std)

	# polynomial feature
	x_train = polynomial_feature(x_train, power = feature_power)


	"Spilt Training and Validation Set"
	if validate_ratio is not 0:
		x_train, y_train, x_val, y_val = validate_split(x_train,y_train, validate_ratio = validate_ratio)

	"Training by Logistic Regression"
	# w_init = np.linalg.inv(x_train.T@x_train)@x_train.T@y_train
	w_init = np.ones((x_train.shape[1],1))
	w_opt, regress_error = grad_descend(x_train,y_train,w_init,iter_num = iter_num, learn_r = learn_rate, optimizer = opt_method)
	# w_opt, regress_error = batch_gradient_descend(x_train, y_train, w_init, batch_size = 100, epoch_num = 200, learn_rate = 0.1)
	" Training and Validation error"
	# predict training set
	y_predict_trian = categorical_predict(x_train,w_opt)
	train_accuracy = categorical_accuracy(y_predict_trian,y_train)
	print('Training Accuracy : %f' %train_accuracy)

	# predict validation set
	if validate_ratio is not 0:
		y_predict_val = categorical_predict(x_val,w_opt)
		val_accuracy = categorical_accuracy(y_predict_val,y_val)
		print('Validation Accuracy : %f' %val_accuracy)


	"Save Training Parameter"
	print('Save training parameter ...')
	model = {}
	model['x_mean'] = x_mean
	model['x_std'] = x_std
	model['w_opt'] = w_opt
	model['feature_power'] = feature_power

	model_name = '../ouput_file/train_model.pickle'
	with open(model_name, 'wb') as write_file :
		pk.dump(model, write_file)



	

