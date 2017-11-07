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
model_path = 'model_cnn_2/'

# training parameter
validation_ratio = 0.3
train_opt = 'cnn'
model_load = False
activate_method = 'relu'
training_num = 28709 # max 28709
batch_size = 32
epoch_num = 50
opt_method = Adadelta(lr = 1)