import tensorflow as tf
import numpy as np
from demo_utils import *

data_path = 'mnist_data/'
row_size = 28
column_size = 28
flatten_size = row_size*column_size
channel = 1
print_opt = True
class_num = 10
batch_size = 1000
epoch_num = 2
iter_number = 1000
img_shape = (row_size,column_size)


def new_weight(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_bias(length):
	return tf.Variable(tf.constant(0.05,shape= [length]))


def new_conv2d_layer(filter_num, ker_size, input, use_pooling = True):
	input_channel = input.get_shape().as_list()[3]
	weights = new_weight([ker_size,ker_size, input_channel, filter_num])
	biases = new_bias(filter_num)

	cnn_output_layer =tf.nn.conv2d( input = input,
									filter = weights,
									strides = [1,1,1,1],
									padding = 'SAME')
	cnn_output_layer += biases

	if use_pooling:
		cnn_output_layer = tf.nn.max_pool(	value=cnn_output_layer, 
											ksize=[1,2,2,1],
											strides = [1,2,2,1],
											padding = 'SAME')
	cnn_output_layer = tf.nn.relu(cnn_output_layer)
	return cnn_output_layer


def flatten_layer(input):
	input_shape = input.get_shape()
	flatten_size = input_shape[1:4].num_elements()
	cnn_flayyen_layer = tf.reshape(input, [-1,flatten_size])
	return cnn_flayyen_layer


def build_dnn_layer(output_size,input,use_relu = True):
	input_size = input.get_shape()[1].value
	weights = new_weight([input_size,output_size])
	biases = new_bias(output_size)
	dnn_output_layer = tf.matmul(input,weights)+biases
	
	if use_relu:
		dnn_output_layer = tf.nn.relu(dnn_output_layer) 
	return dnn_output_layer


def build_cnn_graph(x_train):
	# reshape x_train
	x_train = tf.reshape(x_train, shape = [-1, row_size, column_size, 1])

	# cnn graph
	cnn_output_layer = new_conv2d_layer(filter_num=16,ker_size=5, input=x_train, use_pooling=True)
	cnn_output_layer = new_conv2d_layer(filter_num=32, ker_size=3, input=cnn_output_layer, use_pooling=True)

	# flatten 
	cnn_flayyen_layer = flatten_layer(cnn_output_layer)

	# build fully connected layer
	dnn_output_layer = build_dnn_layer(output_size = 128,input = cnn_flayyen_layer,use_relu = True)
	dnn_output_layer = build_dnn_layer(output_size = 128,input = cnn_flayyen_layer,use_relu = True)
	dnn_output_layer = build_dnn_layer(output_size = class_num,input = cnn_flayyen_layer,use_relu = False)
	logit = tf.nn.softmax(dnn_output_layer)
	return logit

def get_accuracy(y_pred_cls,y_train_cls):
	return tf.reduce_mean(tf.cast(tf.equal(y_pred_cls,y_train_cls),tf.float32))

def optimize_cnn_graph(data, x_train,y_train,y_pred):
	y_pred_cls = tf.argmax(y_pred,dimension = 1)
	y_train_cls = tf.argmax(y_train,dimension = 1)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_train)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
	accuracy = get_accuracy(y_pred_cls,y_train_cls)

	with tf.Session() as sess: 
		sess.run(tf.initialize_all_variables())
		
		for iter in range(iter_number):
			xt_batch,yt_batch = data.train.next_batch(batch_size)
			feed_dict = {x_train:xt_batch,y_train:yt_batch}
			
			sess.run(optimizer, feed_dict = feed_dict)
			loss = sess.run(cost, feed_dict = feed_dict)
			acc = sess.run(accuracy,feed_dict = feed_dict)

			if (iter+1)%5==0:
				print('Loss at iter %d : %.4f\n' %(iter,loss))
				print('Accuracy : %.4f\n' %acc)
		

		x_val = data.validation.images
		y_val = data.validation.labels	
		feed_dict = {x_train:x_val,y_train:y_val}
		acc = sess.run(accuracy,feed_dict = feed_dict)
		print(acc)
	
if __name__ == '__main__':
	"Load data"
	from tensorflow.examples.tutorials.mnist import input_data
	data = input_data.read_data_sets(data_path,one_hot = True)
	data.train.cls = np.argmax(data.train.labels,axis= 1)
	data.test.cls = np.argmax(data.test.labels,axis= 1)
	data.validation.cls = np.argmax(data.validation.labels,axis= 1)
	# plot_images(data.train.images[:9],img_shape,data.train.cls[:9], cls_pred=None)

	if print_opt:
		print("Size of:")
		print("- Training-set:\t\t{}".format(len(data.train.labels)))
		print("- Test-set:\t\t{}".format(len(data.test.labels)))
		print("- Validation-set:\t{}".format(len(data.validation.labels)))

	"place holder"
	x_train = tf.placeholder(tf.float32, shape=[None,flatten_size], name='Xtrain')
	y_train = tf.placeholder(tf.float32, shape=[None,class_num], name='Ytrain')
	
	"build cnn graph"
	y_pred = build_cnn_graph(x_train)

	"optimize on graph"
	optimize_cnn_graph(data,x_train,y_train, y_pred)




