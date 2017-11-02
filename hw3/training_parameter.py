from keras.optimizers import Adadelta
# global parameter
class_list = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
row_size = 48
column_size = 48
channel = 1

class_num = 7
print_opt = True
plot_opt = False

# data path
train_data_path = 'data/train.csv'
model_path = 'model_cnn_1/'

# training parameter
validation_ratio = 0.3
train_opt = 'cnn'
model_load = False
activate_method = 'relu'
training_num = 100 # max 28709
batch_size = 10
epoch_num = 2
opt_method = Adadelta(lr = 1)