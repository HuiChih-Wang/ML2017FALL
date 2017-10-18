# necessary module
import sys
import numpy as np
import pandas as pd
import pickle

np.set_printoptions(suppress = True, precision = 5)

def read_test(test_file_name):
	test_file = pd.read_csv(test_file_name, header = None, usecols = range(2,11))
	test_data = test_file.as_matrix()
	day_num = int(test_data.shape[0]/18)

	feature_hour = np.empty((day_num*9,18))
	idx_hour = 0
	for idx_day in range(day_num):
		data = test_data[range(idx_day*18, (idx_day+1)*18),:]
		
		# replace NR to 0
		rain_bool = data[10,:]=='NR'
		data[10,rain_bool] = u'0'

		for idx in range(9):
			feature_hour[idx_hour,:] = data[:,idx].T 
			idx_hour+=1
	return feature_hour

def data_preprocess (feature_hour,suc_hour = 9):
	day_num = int(feature_hour.shape[0]/9)
	x_test = np.empty((day_num,18*suc_hour))
	for idx_day in range(day_num):
		data = feature_hour[range(idx_day*9, (idx_day+1)*9),:]

		x_test[idx_day,:]=data[-suc_hour:,:].reshape((1,(18*suc_hour))) 
	# add bias term 
	x_test = np.c_[np.ones((day_num,1)), x_test]

	return x_test 

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

def regress_predict(x_train,w):	
	return x_train @ w

if __name__ == '__main__':
	"load training model"
	model_file = 'train_model.pickle'
	with open(model_file, 'rb') as read_file :
		model = pickle.load(read_file)

	x_mean = model['x_mean']
	x_std = model['x_std']
	w_opt = model['w_opt']
	suc_hour = model['suc_hour']
	feature_power = model['feature_power']

	"preprocess raw data"
	test_file_name = 'data/test.csv'
	# data preprocess
	feature_hour = read_test(test_file_name)
	x_test = data_preprocess(feature_hour,suc_hour = suc_hour)

	"prepare feature matrix"
	# feature normalize
	x_test = feature_normalize(x_test, x_mean, x_std)
	# polynomial feature
	x_test = polynomial_feature(x_test, power = feature_power)

	"predict label of test data"
	y_predict = regress_predict(x_test,w_opt)

	# save predict result
	id_list = [ 'id_' + str(i) for i in range(y_predict.shape[0])]
	y_predict_df = pd.DataFrame({'id': id_list, 'value' : y_predict.flatten()}) 

	write_file = 'test_result.csv'
	y_predict_df.to_csv(write_file, index = False)

	