from ReadJsonFiles import ReadFolder
import tensorflow as tf

mypath = 'C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\training\\'

'''
read_obj = ReadFolder(mypath)
read_obj.woof()
x_training, y_training = read_obj.folder_files_to_list()
'''

'''
Get images
'''






'''
Build RandomeForest
'''
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

'''
example with mnist
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = False)

#Parameters
num_steps = 500 #Total steps to train
batch_size = 1024 #Number of samples per batch
num_classes = 10 #The 10 digits
num_features = 784 #28x28 pixels
num_trees = 10
max_nodes = 1000

#Input and Target Data
X = tf.placeholder(tf.float32, shape=[None, num_features])
#For RandomForest labels must be integers (the class ID)
Y = tf.placeholder(tf.int32, shape=[None])

#RandomForest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                        num_features=num_features,
                                        num_trees=num_trees,
                                        max_nodes=max_nodes).fill()

#Build RandomForest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
#get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

#Measure Accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initialize the variables and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

#Start Tensorflow Session
sess = tf.Session()

#run initializer
sess.run(init_vars)

#Training
for i in range(1, num_steps + 1):
    #prepare Data
    #get the next batch of mnist data (images only, no labels)
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X:batch_x, Y:batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X:batch_x, Y:batch_y})
        print('Step %i, Loss %f, Acc: %f' % (i, l, acc))

#Test model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))







#print(read_obj.folder_files_to_list())
#x_training, y_training = read_obj.folder_files_to_list()

#print(x_training[0])
