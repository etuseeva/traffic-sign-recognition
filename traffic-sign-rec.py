import os
import time
import skimage.data
import skimage.transform
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
                      for f in os.listdir(label_dir)]
        # for f in os.listdir(label_dir) if f.endswith(".ppm")]
        labels_str[i] = d
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(i)

        i = i + 1
    return images, labels, labels_str

def test_data(labels_str):
    print('Выберите директорию, в которой хранятся данные для тестирования')
    # test_data_dir = '/home/lena/Desktop/traffic-sign-recognition/images/testing'
    test_data_dir = askdirectory()
    print('Выбранная директория:', test_data_dir)

    print('Загрузка тестовый изображений...')
    start_time = time.time()
    try:
        test_images, test_labels, test_labels_str = load_data(test_data_dir)
    except Error:
        print('Ошибка в момент загрузки данных.')
        return
    print('Загрузка изображений была успешно завершена за ', time.time() - start_time, ' милисекунд')

    start_time = time.time()
    print('Производится сжатие изображений до 32x32 пикселя...')
    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
    print('Сжатие изображений завершено за ', time.time() - start_time, ' милисекунд')

    # for i in predicted:
    # print(labels_str[i])

    print('Проверка правдивости распознавателя:')
    sum = 0
    for i in range(len(predicted)):
        if predicted[i] == test_labels[i]:
            sum = sum + 1

    print('Правдивость равна = ', (sum / len(test_labels)) * 100, '%')

def run_on_image(labels_str):
    print('Выберите изображение которое хотите распознать: ')
    image_path = askopenfilename()
    print('Путь до изображения:', image_path)

    test_images = []
    test_images.append(skimage.data.imread(image_path))

    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]

    # for i in predicted:
    #     print(labels_str[i])

    for i in range(len(predicted)):
        plt.subplot(1, 1, i + 1)
        plt.text(0, -5, "It's - {0}".format(labels_str[predicted[i]]),
                 fontsize=12, color='black')
        plt.imshow(test_images[i])
        plt.axis('off')
    plt.show()

print('Выберите директорию, в которой хранятся тренировочные изображения.')
# train_data_dir = '/home/lena/Desktop/traffic-sign-recognition/images/training'
train_data_dir = askdirectory()
print('Выбранная директория: ', train_data_dir)

print('Проводится загрузка предложенных данных...')
start_time = time.time()
try:
    images, labels, labels_str = load_data(train_data_dir)
except Error:
    print('Ошибка в момент загрузки данных.')
    return
print('Загрузка изображений была успешно завершена за ', time.time() - start_time, ' милисекунд')
print('Всего загружено ', len(images), ' изображений')

amount_images = len(set(labels))

start_time = time.time()
print('Производится сжатие изображений до 32x32 пикселя...')
images32 = [skimage.transform.resize(image, (32, 32)) for image in images]
print('Сжатие изображений завершено за ', time.time() - start_time, ' милисекунд')

images_a = np.array(images32)
labels_a = np.array(labels)

print('Создание графа.')
graph = tf.Graph()

print('Введите параметры обучения:')
learning_rate = float(input("Скорость обучения (learning rate): "))
learning_steps = int(input("Количество итераций обучения: "))

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

print('Создание сессии.')
start_time = time.time()
print('Производится обучение...')
for i in range(learning_steps):
    _, loss_value = session.run([train, loss],
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
print('Обучение завершено за ', time.time() - start_time, ' милисекунд')

while True:
    print('Что вы хотите сделать? ' +
          '1: Проверить прадивость распознавателя на тренировочных данных. \n' +
          '2: Распознать изображения; \n'
          '3: Выход.')
    type = int(input("Введите число: "))

    if type == 1 :
        test_data(labels_str)
    elif type == 2:
        while True:
            run_on_image(labels_str)
            ans = input("Продолжить распознавания изображений? y/n ")
            if ans == 'n':
                break
    elif type == 3:
        break

session.close()