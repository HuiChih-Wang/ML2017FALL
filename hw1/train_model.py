# necessary module
import sys
import numpy as np
import pandas as pd
import pickle


# training parameter
suc_hour = 5
feature_power = 2
validate_ratio = 0.3
learning_rate = 1
regular_par = 1
optimize_method = 'gradient_descend'
np.set_printoptions(suppress = True, precision = 5)


#  function code
def read_train(train_file_name,days = 240):
	feature_range = list(range(3,27)) 
	train_file = pd.read_csv(train_file_name,nrows = 18*days, encoding = 'ISO-8859-1', usecols = feature_range)
	train_file = train_file.as_matrix()
	
	feature_hour = np.zeros((days*24, 18))
	data_count = 0
	for idx in range(days):
		row_idx = np.arange(18*idx,18*(idx+1))
		train_day = train_file[row_idx,:]

		# replace 'NR' to '0'
		rain_bool = train_day[10,:]=='NR'
		train_day[10,rain_bool] = u'0'

		train_day = train_day.astype(float)
		for i in range(24):
			feature_hour[[data_count],:] = train_day[:,[i]].T
			data_count +=1

	return feature_hour

def data_preprocess(feature_hour, suc_hour = 9):
	data_num = feature_hour.shape[0]-suc_hour
	feature_num = 18 * suc_hour

	# split label
	y_hour = feature_hour[:,[9]]
	
	# feature matrix with successive hour suc_hour
	x_train = np.empty((data_num,feature_num))
	y_train = np.empty((data_num,1))
	for idx in range(data_num):
		feature = feature_hour[idx:idx+suc_hour,:]
		x_train[[idx],:] = feature.reshape((1,feature_num))
		y_train[idx,:] = y_hour[idx+suc_hour,:]

	# add bias term 
	x_train = np.c_[np.ones((data_num,1)), x_train]
	return x_train,y_train	

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


def mse_error(x_train,y_train,w):
	data_num = x_train.shape[0]
	diff = x_train @ w - y_train
	w_regular = w[1:,:]
	err = np.mean(diff**2) + 0.5*regular_par*np.sum(w_regular**2)/data_num
	return err

def mse_gradient(x_train, y_train, w):
	data_num = x_train.shape[0]
	diff = x_train @ w - y_train
	E = np.identity(x_train.shape[1])
	E[0,0] = 0
	w_regular = E * w
	grad = (2 * x_train.T @ diff + regular_par * w_regular)/data_num
	return grad

def grad_descend(x_train, y_train ,w_init, iter_num = 1000, learn_r =0.1, optimizer = 'normal'):

	w = w_init 
	regress_error = np.empty((iter_num,1))
	
	if optimizer is 'ada_grad':
		grad_prev = mse_gradient(x_train,y_train,w_init)
		for iter in range(iter_num):
			# ada grad
			grad_prev_rms = np.sqrt(np.sum(grad_prev**2)/grad_prev.shape[1])
			grad = mse_gradient(x_train, y_train, w)
			w = w - learn_r * grad/grad_prev_rms

			grad_prev = np.c_[grad_prev, grad]
			regress_error[iter,:] = mse_error(x_train,y_train,w)
			# print(regress_error[iter])

	elif optimizer is 'normal' : 
		for iter in range(iter_num):
			grad = mse_gradient(x_train,y_train,w)
			grad_norm = np.linalg.norm(grad)
			w = w - learn_r*grad/grad_norm
			regress_error[iter,:] = mse_error(x_train,y_train,w)
			# print(regress_error[iter])

	return w, regress_error
def get_weight_opt(x_train,y_train):
	E = np.identity(x_train.shape[1])
	E[0,0] = 0
	w_opt = np.linalg.inv(x_train.T @ x_train + 0.5*regular_par*E) @ x_train.T @ y_train
	return w_opt

def regress_predict(x_train,w):	
	return x_train @ w


# main code
if __name__ == '__main__':
	train_file = 'data/train.csv'
	feature_hour = read_train(train_file, days = 240)

	"data preprocessing"
	print('successive hour :% d  Feature power : %d' %(suc_hour, feature_power))
	# data preprocessing
	x_train, y_train = data_preprocess(feature_hour, suc_hour = suc_hour)

	"prepare feature matrix"
	# feature normalization
	x_mean = np.mean(x_train, axis = 0)
	x_std = np.std(x_train, axis = 0)
	x_train = feature_normalize(x_train,x_mean,x_std)

	# polynomial feature
	x_train = polynomial_feature(x_train, power = feature_power)

	"split trian and validate set"
	if validate_ratio is not 0:
		x_train, y_train, x_val, y_val = validate_split(x_train,y_train, validate_ratio = validate_ratio)

	"training by gradient descend algorithm to find optimal w"
	# set initial weight
	w_init = np.ones((x_train.shape[1],1))
	
	if optimize_method is 'gradient_descend':
	# gradient descend
		if validate_ratio is 0:
			print('Train best model by gradient descend...')
		else:
			print('Training and Validation by gradient descend with valitation ratio %.2f ...' %validate_ratio)

		w_opt, regress_error = grad_descend(x_train,y_train,w_init,iter_num = 5000,learn_r = learning_rate, optimizer = 'normal')
		print(regress_error[-20:])

	elif optimize_method is 'close_form':
		print('Training and Validation by close form solution with validation ratio %.2f ...' %validate_ratio)
		w_opt = get_weight_opt(x_train,y_train)


	" training and validation error"
	# predict training set
	y_predict_trian = regress_predict(x_train,w_opt)
	train_error = mse_error(x_train,y_train,w_opt)
	print('Training Error : %f' %train_error)

	# predict validation set
	if validate_ratio is not 0:
		y_predict_val = regress_predict(x_val,w_opt)
		val_error = mse_error(x_val,y_val,w_opt)
		print('Validation Error : %f' %val_error)

	"save training model"
	print('Save training parameter ...')
	model = {}
	model['x_mean'] = x_mean
	model['x_std'] = x_std
	model['suc_hour'] = suc_hour
	model['w_opt'] = w_opt
	model['feature_power'] = feature_power

	model_name = 'train_model.pickle'
	with open(model_name, 'wb') as write_file :
		pickle.dump(model, write_file)








