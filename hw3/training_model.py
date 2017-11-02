import sys
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization 
from keras.losses import categorical_crossentropy

from training_parameter import *


def build_cnn(input_shape):
	cnn_model = Sequential()
	cnn_model.add(Conv2D(32, (3,3), activation = activate_method, padding = 'same', input_shape = input_shape))
	cnn_model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	# cnn_model.add(Dropout(0.25))
	cnn_model.add(Conv2D(64, (3,3), activation = activate_method, padding = 'same'))
	cnn_model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	cnn_model.add(Conv2D(128, (3,3), activation = activate_method, padding = 'same'))
	cnn_model.add(BatchNormalization())
	cnn_model.add(MaxPool2D(pool_size = (2,2)))
	# cnn_model.add(Dropout(0.25))

	# flatten dense layer
	cnn_model.add(Flatten())
	cnn_model.add(Dense(256, activation = 'relu'))
	cnn_model.add(Dropout(0.25))
	cnn_model.add(Dense(256, activation = 'relu'))
	cnn_model.add(Dropout(0.25))
	cnn_model.add(Dense(256, activation = 'relu'))
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

def train_model(x_train, y_train, x_val, y_val):
	# build model
	if model_load:
		print('Load %s model from path: %s\n' %(train_opt,model_path))
		model = load_model(model_path)
	else:
		print('Traing %s model with %d traing data and %d validation data' %(train_opt, x_train.shape[0],x_val.shape[0]))
		if train_opt == 'cnn':
			model = build_cnn(input_shape = x_train.shape[1:])
		elif train_opt == 'dnn': 
			model = build_dnn(input_shape = x_train.shape[1:])
		else:
			print("Error!Default model option:'cnn','dnn'\n")
			sys.exit()
		# print model summary
		if print_opt:
			model.summary()
		# fit model
		model.compile(loss = categorical_crossentropy, optimizer = opt_method, metrics = ['accuracy'])
		model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch_num, validation_data = (x_val,y_val), verbose = int(print_opt))
		# save out model
		print('Save %s  model to path:%s\n' %(train_opt,model_path))
		model.save(model_path)
	return model