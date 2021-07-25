import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
import numpy as np

convnet = input_data(shape=[None, 150, 150, 3], name='input')

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = avg_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 1024, 6, activation='relu')
convnet = avg_pool_2d(convnet, 6)

convnet = dropout(convnet, 0.25)

convnet = conv_2d(convnet, 1024, 6, activation='relu')
convnet = avg_pool_2d(convnet, 6)

convnet = dropout(convnet, 0.25)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = avg_pool_2d(convnet, 5)
convent = flatten(convnet)
convnet = fully_connected(convnet, 128, activation='relu')
convnet = fully_connected(convnet, 256, activation='relu')
convnet = fully_connected(convnet, 6, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=1e-3,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load('[NSR]CNN.tfl')

test = []
for img in os.listdir('testData'):
  path = os.path.join('testData', img)
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (150, 150))
  test.append(np.array(img))
test = np.array(test)
test = test.reshape(-1,150, 150, 3)

prediction = model.predict(test)
label = ['buildings','forest','glacier','mountain','sea','street']
i = 0
for p in prediction:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(test[i])
    plt.title(label[np.argmax(p)])
    plt.show()
    print(label[np.argmax(p)])
    i += 1

