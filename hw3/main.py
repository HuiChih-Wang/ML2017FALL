import numpy as np
from training_parameter import*
from training_model import train_model
from training_utils import *


if __name__ == '__main__':
	"Load image data"
	x_train, y_train = load_image()
	print(training_num)
	training_num = y_train.shape[0]

	# validation split
	(x_train,y_train), (x_val,y_val) = validation_split(x_train,y_train)

	# class y_train y_val
	y_val_class = np.argmax(y_val,axis = 1)
	y_train_class = np.argmax(y_train,axis = 1)


	"Build training model" 
	# y_train_weight = sample_weight(y_train)
	# y_train_weight = None
	model = train_model(x_train, y_train, x_val, y_val)


	"Evaluate accuracy"
	y_val_predict = model.predict_classes(x_val)
	y_train_predict = model.predict_classes(x_train)
	train_acc = get_accuracy(y_true = y_train_class, y_predict = y_train_predict)
	val_acc = get_accuracy(y_true = y_val_class, y_predict = y_val_predict)


	if print_opt: 
		print("\n\nAccuracy Evaluation")
		print("Training accuracy: %4f\n" %train_acc)
		print("Validation accuracy: %4f\n" %val_acc)

	"Confusion matrix"
	if plot_opt:  
		cm = get_confusion_matrix(y_true = y_val_class,y_predict = y_val_predict)
		plot_confusion_mat(cm)
