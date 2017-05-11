import os
import time
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tkinter.filedialog import *

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    labels_str = {}
    images = []

    i = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
                      # for f in os.listdir(label_dir) if f.endswith(".jpg")]
        labels_str[i] = d
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(i)
            
        i = i + 1
    return images, labels, labels_str


def test_data(labels_str):
    print('Select directory with testing data')
    # test_data_dir = '/home/lena/Desktop/traffic-sign-recognition/images/testing'
    test_data_dir = askdirectory()
    print('Testing directory:', test_data_dir)

    print('Loading testing images...')
    start_time = time.time()
    try:
        test_images, test_labels, test_labels_str = load_data(test_data_dir)
    except Error:
        print('Testing data - error loading')
    print('Loading data has ended: ', time.time() - start_time)

    start_time = time.time()
    print('Running resize images')
    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels], 
                            feed_dict={images_ph: test_images32})[0]
    print('Resizing images has ended...: ', time.time() - start_time)
    
    # for i in predicted:
        # print(labels_str[i])

    print('Check truthfulness graph...')
    sum = 0
    for i in range(len(predicted)):
        if predicted[i] == test_labels[i]:
            sum = sum + 1

    print('Truthfulness: ', (sum / len(test_labels)) * 100, '%');


def run_on_image(labels_str):
    print('Select image')
    image_path = askopenfilename()
    print('Image path:', image_path)
    
    test_images = []    
    test_images.append(skimage.data.imread(image_path))
    
    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels], 
                            feed_dict={images_ph: test_images32})[0]

    for i in predicted:
        print(labels_str[i])

    for i in range(len(predicted)):
        plt.subplot(1, 1, i + 1)
        plt.text(0, -5, "It's - {0}".format(labels_str[predicted[i]]), 
                     fontsize=12, color='black')
        plt.imshow(test_images[i])
        plt.axis('off')
    plt.show()


print('Select folder with training images')
# train_data_dir = '/home/lena/Desktop/traffic-sign-recognition/images/training'
train_data_dir = askdirectory()
print('Training directory:', train_data_dir)

print('Loading training images...')
start_time = time.time()
try:
    images, labels, labels_str = load_data(train_data_dir)
except Error:
    print('Training data - error loading')
print('Loading data has ended: ', time.time() - start_time)
print('Load ', len(images), 'images')

amount_images = len(set(labels))

start_time = time.time()
print('Running resize images')
images32 = [skimage.transform.resize(image, (32, 32)) for image in images]
print('Resizing images has ended...: ', time.time() - start_time)

images_a = np.array(images32)
labels_a = np.array(labels)

print('Graph has created...')
graph = tf.Graph()

with graph.as_default():
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    images_flat = tf.contrib.layers.flatten(images_ph)
    logits = tf.contrib.layers.fully_connected(images_flat, amount_images, tf.nn.relu)
    predicted_labels = tf.argmax(logits, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph, 
        logits=logits))
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)
_ = session.run([init])

print('Session has created...')
start_time = time.time()
print('Running train...')
for i in range(200):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
print('Train has ended: ', time.time() - start_time)

# Testing
test_data(labels_str)

# run_on_image(labels_str)

session.close()