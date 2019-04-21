import json
import os
import base64
from labelme import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import color
from skimage import io

from os import listdir
from os.path import isfile, join

class ReadFolder:
    def woof(self):
        print("sup")
    def __init__(self, path):
        self.path = path

    def bark(self):
        print("wassup")

    def folder_files_to_list(self):
        print(self.path)
        files = [self.path + f for f in listdir(self.path) if isfile(join(self.path, f))]
        x = []
        y = []
        for file in files:
            with open(file) as json_file:
                data = json.load(json_file)
                x.append(utils.img_b64_to_arr(data['imageData']))
                y.append(data['shapes'][0]['label'])
        return x, y

        def to_grayscale(pictures):
            gray_pictures
            for x in pictures:
                gray_pictures.append(color.rgb2gray(x))
            return gray_pictures
'''
mypath = 'C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\Project Data\\Labeled France Potholes\\'
onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]

print(len(onlyfiles))

with open('C:\\Users\jross\OneDrive\Desktop\\yo.txt') as json_file:
    data = json.load(json_file)

with open(onlyfiles[1]) as json_file:
    data = json.load(json_file)

mypath = 'C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\Project Data\\Labeled France Potholes\\'
onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]


imageData = data['imageData']

img = utils.img_b64_to_arr(imageData)

x_training = []
y_training = []

print(data['shapes'][0]['label'])

for file in onlyfiles:
    with open(file) as json_file:
        data = json.load(json_file)
        x_training.append(utils.img_b64_to_arr(data['imageData']))
        y_training.append(data['shapes'][0]['label'])
'''


'''
paramter: path to the folder containing json files
works for labelme files
returns: 2 lists, one with jpeg and one with labels
'''

'''
x_trainging1, y_training1 = folder_files_to_list('C:\\Users\\jross\\OneDrive\\Data Analytics\\Machine Learning\\Project\\Project Data\\Labeled France Potholes\\')
print(x_trainging1[0])
'''
'''
print(x_training[0])
print(y_training[0])

print(x_training[0].shape)

#convert to gray-scale
from skimage import color
from skimage import io

print(x_training[0][:,:,0])
x_training_gray = []
'''
'''
accepts a list of pictures
converts list into grayscale
returns the list
'''
'''
def to_grayscale(pictures):
    gray_pictures
    for x in pictures:
        gray_pictures.append(color.rgb2gray(x))
    return gray_pictures
    '''
'''
#print(x_training_gray.shape[0])
print(x_training_gray[0])
#print(x_training_gray)
io.imshow(x_training_gray[0])
io.show()
'''
