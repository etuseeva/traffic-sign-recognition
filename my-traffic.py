import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

amount_images = 3

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label: {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

# Dict
str_labels = {
    0: 'no parking',
    1: 'no entry',
    2: 'stop',
}

# -------------
# Todo: Select path with dialog window
ROOT_PATH = "/home/lena/Desktop/course_work/images"
train_data_dir = os.path.join(ROOT_PATH, "training")
test_data_dir = os.path.join(ROOT_PATH, "testing")

# ROOT_PATH = "/home/lena/Desktop/course_work/traffic"
# train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
# test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)

# -------------
# Resize images
images32 = [skimage.transform.resize(image, (32, 32)) for image in images]

images_a = np.array(images32)
labels_a = np.array(labels)

# -------------
# Create graph
graph = tf.Graph()

with graph.as_default():
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    images_flat = tf.contrib.layers.flatten(images_ph)
    logits = tf.contrib.layers.fully_connected(images_flat, amount_images, tf.nn.relu)
    predicted_labels = tf.argmax(logits, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph, 
        logits=logits))
    train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)
_ = session.run([init])

for i in range(200):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})

# while (True): 
#     sample_indexes = random.sample(range(len(images32)), 1)
#     print('sample_indexes = ', sample_indexes)

#     sample_images = [images32[i] for i in sample_indexes]
#     sample_labels = [labels[i] for i in sample_indexes]

#     predicted = session.run([predicted_labels], 
#                             feed_dict={images_ph: sample_images})[0]
#     print(sample_labels)
#     print(predicted)

#     predicted = session.run([predicted_labels], feed_dict={images_ph: sample_images})[0]
    
#     fig = plt.figure(figsize=(10, 10))
#     for i in range(len(sample_images)):
#         print('i = ', i)
#         truth = sample_labels[i]
#         print('truth = ', truth)
#         prediction = predicted[i]
#         print('prediction = ', prediction)

#         plt.subplot(1, 2, i + 1)
#         plt.axis('off')

#         color='green' if truth == prediction else 'red'
#         plt.text(40, 10, 'Prediction: {0}'.format(str_labels[prediction]), 
#                  fontsize=12, color=color)

#         plt.imshow(sample_images[i])

#         # plt.subplot(1, 2, i + 2)
#         # plt.axis('off')
#         # plt.imshow(images32[sample_indexes[i]])

#     plt.show()

#     inp = input("Next?\n")
#     if inp != "yes":
#         break

# ---------
# Test data
test_images, test_labels = load_data(test_data_dir)
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]

predicted = session.run([predicted_labels], feed_dict={images_ph: test_images32})[0]
print(predicted)

for i in range(len(predicted)):
    plt.subplot(5, 2, i + 1)
    plt.axis('off')
    plt.text(40, 10, "Prediction: {0}".format(str_labels[predicted[i]]), 
                 fontsize=12, color='black')
    plt.imshow(test_images[i])

    # plt.subplot(1, 2, i + 2)
    # plt.axis('off')
    # plt.imshow(images32[sample_indexes[i]])
plt.show()

session.close()