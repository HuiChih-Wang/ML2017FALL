import tensorflow as tf
import numpy as np
import os

data_path = 'mnist_data/'
row_size = 28
column_size = 28
flatten_size = row_size*column_size
channel = 1
print_opt = True
class_num = 10
batch_size = 1000
epoch_num = 2


def build_dnn_graph(data):
	x_train = tf.placeholder(tf.float32, [None,flatten_size],name = 'Xtrain')
	y_true = tf.placeholder(tf.float32, [None, class_num], name = 'Ytrue')
	y_true_cls = tf.argmax(y_true, dimension = 1)

	w = tf.Variable(tf.zeros([flatten_size,class_num]), name = 'W1')
	b = tf.Variable(tf.zeros([class_num]), name = 'b1')
	h = tf.matmul(x_train,w) + b
	y_prob = tf.nn.softmax(h)
	y_pred_cls = tf.argmax(y_prob, dimension = 1)

	# optimization
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_prob,labels=y_true)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1).minimize(cost)

	# performance measure
	cls_equal = tf.cast(tf.equal(y_pred_cls,y_true_cls),tf.float32)
	accuracy = tf.reduce_mean(cls_equal)

	# variable initialization
	init = tf.global_variables_initializer()

	# training process
	with tf.Session() as sess:
		sess.run(init)
		for epoch_idx in range(epoch_num):
			for iter in range(int(55000/batch_size)):
				x_batch, y_true_batch = data.train.next_batch(batch_size)
				# print(x_batch.shape,y_true_batch.shape)
				feed_dict = {x_train: x_batch, y_true:y_true_batch}
				sess.run(optimizer, feed_dict = feed_dict)
				loss = sess.run(cost, feed_dict = feed_dict)
				acc = sess.run(accuracy,feed_dict = feed_dict)
				if (iter+1)%5==0:
					print('Loss at iter %d : %.4f\n' %(iter,loss))
					print('Accuracy : %.4f\n' %acc)

			x_batch, y_true_batch = data.train.next_batch(55000)
			feed_dict = {x_train: x_batch, y_true:y_true_batch}
			acc = sess.run(accuracy,feed_dict = feed_dict)
			print('Accuracy in epoch %d : %.4f\n\n\n' %(epoch_idx,acc))





# def run_tf_session():
# 	sess = tf.Session()
# 	sess.run(init)







	sess.close()


if __name__ == '__main__':
	"Load data"
	from tensorflow.examples.tutorials.mnist import input_data
	data = input_data.read_data_sets(data_path,one_hot = True)
	data.train.cls = np.argmax(data.train.labels,axis= 1)
	data.test.cls = np.argmax(data.test.labels,axis= 1)
	data.validation.cls = np.argmax(data.validation.labels,axis= 1)

	if print_opt:
		print("Size of:")
		print("- Training-set:\t\t{}".format(len(data.train.labels)))
		print("- Test-set:\t\t{}".format(len(data.test.labels)))
		print("- Validation-set:\t{}".format(len(data.validation.labels)))


	"Build computational graph"
	build_dnn_graph(data)

	"Run tensorflow section"



	