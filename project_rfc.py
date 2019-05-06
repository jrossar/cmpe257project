import numpy as np
import json
from labelme import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import cv2
from os import listdir
from os.path import isfile, join

from skimage import color
from skimage import io

mypath = 'C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\Project Data\\Labeled France Potholes\\'
DATADIR = "C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\training"
DATADIR2 = "C:\\Users\\jross\\Downloads\\datasets"
count = 0

import os

path = '.'
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

def create_training_data(path):
    training_data = []
    training_labels = []
    for file in os.listdir(path):
        file_array = []
        for file2 in os.listdir(os.path.join(path,file)):
            file_array.append(file2)
            if(file2=="label_names.txt"):
                with open(os.path.join(path,file,file2)) as f:
                    data_per_file = []
                    next(f)
                    for line in f:
                        data_per_file.append(line)
                    training_labels.append(data_per_file)
            if(file2 =="label_viz.png"):
                    img_array = cv2.imread(os.path.join(path,file,file2),0)
                    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                    training_data.append([new_array])
        if 'label_names.txt' not in file_array:
            print(file)
    return training_data, training_labels

# training_data, training_labels, count = folder_files_to_list(mypath)
training_data, training_labels = create_training_data(DATADIR)
training_data_2, training_labels_2 = create_training_data(DATADIR2)
print(len(training_data_2))
training_data.extend(training_data_2)
print(len(training_data))
training_labels.extend(training_labels_2)
print(len(training_data), len(training_labels))
np_training_data = np.array(training_data)
np_training_labels = np.array(training_labels)
print(np_training_data.shape)

#np_training_labels = np_training_labels.reshape(len(training_labels),)

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

np_training_data = np_training_data[:,0,:,:]
np_training_data = np_training_data.reshape(len(np_training_data), 784)

def to_grayscale(pictures):
    gray_pictures = []
    for x in pictures:
        gray_pictures.append(color.rgb2gray(x))
        return gray_pictures


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=17)

#print([s.strip('\\n') for s in np_training_data])

'''
for i in range(0, len(class_num)):
    for j in range(0, len(class_num[i])):
        class_num[i][j] = class_num[i][j].strip()
'''

#np_training_labels = np.array(class_num)

'''
for i in range(0, len(np_training_labels)):
    np_training_labels[i] = np.array(np_training_labels[i])
    np_training_labels[i] = np.array(np_training_labels[i].astype('int'))



list_thing = []
for i in range(0, 236):
    list_thing.append([0,1,0,1])
'''

for i in range(0, len(class_num)):
    for j in range(0, len(class_num[i])):
        class_num[i][j] = int(class_num[i][j])

np_classes = np.array(training_labels)
#one_hot_targets = np.eye(3)[np_classes]
#print(one_hot_targets)

'''
one_hot = []

for i in range(0, len(training_labels)):
    temp = []
    if 1 in class_num[i]:
        temp.append(1)
    else:
        temp.append(0)
    if 2 in class_num[i]:
        temp.append(1)
    else:
        temp.append(0)
    if 3 in class_num[i]:
        temp.append(1)
    else:
        temp.append(0)
    one_hot.append(temp)

from sklearn.preprocessing import OneHotEncoder
np_classes = np.array(class_num)
#np_classes = np_classes.reshape(236, 3)
'''

for i in range(0, len(np_classes)):
    np_classes[i] = np.array(np_classes[i])

def shape_array(array, max_len):
    for i in range(0, len(array)):
        while(len(array[i]) < max_len):
            array[i] = np.append(array[i], [0])
    return array

np_classes = shape_array(np_classes, 3)

print(np_classes[0][0])
print(np_classes.shape)

som = np.array([[1, 2],[3, 4]])
print(som)
print(type(som), type(som[1]), type(som[1][1]))
print(som.shape)
print(type(np_classes), type(np_classes[1]), type(np_classes[1][1]))

#print(type(list_thing[0][0]))

#random_forest.predict(np_training_data)
#random_forest.score(np_training_data, one_hot)

from sklearn.preprocessing import MultiLabelBinarizer

def remove_char(data):
    for d in data:
        for i in range(0, len(d)):
            d[i] = d[i].rstrip('\n')

remove_char(training_labels)

mlb = MultiLabelBinarizer(classes=['0', '1','2','3'])
mlb_label_train = mlb.fit_transform(training_labels)

random_forest.fit(np_training_data, mlb_label_train)
print("done")

print(random_forest.score(np_training_data, mlb_label_train))
print(mlb.classes_)

print(np_training_data.shape)
print(np_training_data[0].shape)
print(random_forest.predict([np_training_data[0]]))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(random_forest, np_training_data, mlb_label_train, scoring='accuracy', cv=10)

print('scores:', scores)
#precision = cross_val_score(random_forest, np_training_data, mlb_label_train, scoring='average_precision', cv=10)
#print(precision)

TestDir = "C:\\Users\\jross\\Downloads\\TrainingSet257"

test_data, test_labels = create_training_data(TestDir)

remove_char(test_labels)

mlb_test = MultiLabelBinarizer(classes=['1','2','3'])
mlb_label_test = mlb.fit_transform(test_labels)

print("Test Score:", random_forest.score(test_data, mlb_label_test))
