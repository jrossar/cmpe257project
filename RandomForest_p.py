import numpy as np
import json
from labelme import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import cv2
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

from skimage import color
from skimage import io

mypath = 'C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\Project Data\\Labeled France Potholes\\'
DATADIR = "C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\training"
count = 0 ;

import os

path = '.'
training_data = []
training_labels = []
class_num = []
IMG_SIZE = 28

def folder_files_to_list(path):
    print(path)
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    x = []
    y = []
    labels_count = []
    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            x.append(utils.img_b64_to_arr(data['imageData']))
            y.append(data['shapes'][0]['label'])
            labels_count.append(len(data['shapes']))
    return x, y, labels_count

def create_training_data():
    for file in os.listdir(DATADIR):
        for file2 in os.listdir(os.path.join(DATADIR,file)):
            if(file2=="label_names.txt"):
                with open(os.path.join(DATADIR,file,file2)) as f:
                    data_per_file = []
                    next(f)
                    for line in f:
                        data_per_file.append(line)
                    class_num.append(data_per_file)
            if(file2 =="label_viz.png"):
                    img_array = cv2.imread(os.path.join(DATADIR,file,file2),0)
                    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                    training_data.append([new_array])

# training_data, training_labels, count = folder_files_to_list(mypath)
create_training_data()
np_training_data = np.array(training_data)
np_training_labels = np.array(class_num)
np_training_labels = np_training_labels.reshape(236,)

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
print(len(np_training_labels))
print(len(np_training_data))
np_training_data = np_training_data[0:236,:]
print(len(np_training_data))

print("ONEEEEEEEEE")
np_training_data = np_training_data[:,0,:,:]
np_training_data = np_training_data.reshape(236, 784)
print("TWOOOOOOOOOO")
# Input and Target data
X = tf.placeholder(tf.float32, shape=[236])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[236])
print("THREEEEEEE")
# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)
print("FOURRRR")
# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))
print("FIVEEEEEEE")
# Start TensorFlow session
sess = tf.Session()
print("SIXx")
# Run the initializer
sess.run(init_vars)
print("SIXXXXXXXX")

print(np_training_labels.shape)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    print(i)
    batch_x = np_training_data[i]
    batch_y = np_training_data[i]
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    print(i)
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: bath_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))


def to_grayscale(pictures):
    gray_pictures
    for x in pictures:
        gray_pictures.append(color.rgb2gray(x))
        return gray_pictures
