# imported module
import sys
import numpy as np
import pandas as pd
import pickle as pk
np.set_printoptions(precision = 5, suppress = True)


# function code
def read_csv(test_file_name):
	x_test = pd.read_csv(test_file_name).as_matrix()
	# add bias term 
	x_test = np.c_[np.ones((x_test.shape[0],1)), x_test]
	return x_test


def feature_normalize(x_train, train_mean, train_std):
	x_train_normalize = np.empty(x_train.shape)
	for idx in range(x_train.shape[1]): 
		if train_std[idx] != 0:
			x_train_normalize[:,[idx]] = (x_train[:,[idx]]-train_mean[idx])/train_std[idx]
		else:
			x_train_normalize[:,[idx]] = x_train[:,[idx]]
	return x_train_normalize


def polynomial_feature(x_train, power = 3):
	x_train_poly = x_train
	for idx in range(2,power+1):
		x_train_poly = np.c_[x_train_poly, x_train[:,1:]**idx]
	return x_train_poly


def sigmoid(value):
	return 1/(1+np.exp(-value))


def prob_predict(x_train, w):
	return sigmoid(x_train @ w)

def categorical_predict(x_train,w):
	f = prob_predict(x_train,w)
	y_predict = np.zeros((x_train.shape[0],1))
	y_predict[f>=0.5] = 1
	return y_predict

# main code
if __name__ == "__main__":
	test_file_name = '../data/X_test'
	x_test = read_csv(test_file_name)

	# load traing model
	train_model_name = '../ouput_file/train_model.pickle'
	with open(train_model_name,'rb') as model_file:
		model = pk.load(model_file)

	# feature normalization
	x_mean = model['x_mean']
	x_std = model['x_std']
	x_test = feature_normalize(x_test,x_mean, x_std)
		
	# polynomial feature
	feature_power = model['feature_power']
	x_test = polynomial_feature(x_test,power = feature_power)

	# predict test predict label
	w_opt = model['w_opt']
	y_predict = categorical_predict(x_test,w_opt)


	# save predict result
	id_list = range(1,y_predict.shape[0]+1)
	y_predict_df = pd.DataFrame({'id': id_list, 'value' : y_predict.flatten().astype(int)}) 

	write_file = '../ouput_file/test_result.csv'
	y_predict_df.to_csv(write_file, index = False)
