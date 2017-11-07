import numpy as np
from training_parameter import *
from training_model import train_model
from training_utils import *
# import matplotlib.pyplot as plt

	

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i,:,:,0],cmap ='gray')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

if __name__ == '__main__':
	"Load image data"
	x_train, y_train = load_image()
	training_num = y_train.shape[0]
	y_train_cls = np.argmax(y_train,axis = 1)
	
	# validation split
	(x_train,y_train), (x_val,y_val) = validation_split(x_train,y_train)

	# class y_train y_val
	y_val_class = np.argmax(y_val,axis = 1)
	y_train_class = np.argmax(y_train,axis = 1)


	"Build training model" 
	# y_train_weight = sample_weight(y_train)
	# y_train_weight = None
	model = train_model(x_train, y_train, x_val, y_val,train_by_generator = True)


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

	"Write simulation parameter"
	write_simulation_text()