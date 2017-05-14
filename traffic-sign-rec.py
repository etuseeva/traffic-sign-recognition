import os
import time
import skimage.data
import skimage.transform
import matplotlib as mpl
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
                      for f in os.listdir(label_dir)]
        labels_str[i] = d
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(i)
        i = i + 1
    return images, labels, labels_str

def test_data(labels_str):
    print('Введите директорию, в которой хранятся данные для тестирования:')
    print('/home/lena/Desktop/traffic-sign-recognition/Знаки ПДД/Тестовые данные')
    test_data_dir = '/home/lena/Desktop/traffic-sign-recognition/Знаки ПДД/Тестовые данные'
    test_data_dir = askdirectory()
    print('Выбранная директория:', test_data_dir)

    print('Загрузка тестовый изображений...')
    start_time = time.time()
    test_images, test_labels, test_labels_str = load_data(test_data_dir)
    print('Загрузка изображений была успешно завершена за ', time.time() - start_time, ' милисекунд')
    print('Всего загружено ', len(test_images), ' изображений')

    start_time = time.time()
    print('Производится сжатие изображений до 32x32 пикселя...')
    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
    print('Сжатие изображений завершено за ', time.time() - start_time, ' милисекунд')

    print('Проверка правдивости распознавателя:')
    sum = 0
    for i in range(len(predicted)):
        if predicted[i] == test_labels[i]:
            sum = sum + 1

    print('Правдивость равна = ', (sum / len(test_labels)) * 100, '%')
    return

def run_on_image(labels_str):
    print('Выберите путь до изображения которое хотите распознать: ')
    image_path = askopenfilename()
    print(image_path)
    test_images = []
    test_images.append(skimage.data.imread(image_path))

    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
    for i in range(len(predicted)):
        plt.subplot(1, 1, i + 1)
        print(labels_str[predicted[i]])
        plt.text(0, -5, labels_str[predicted[i]], fontsize=18, color='black')
        plt.imshow(test_images[i])
        plt.axis('off')
    return

mpl.rcParams['font.family'] = 'fantasy'
mpl.rcParams['font.fantasy'] = 'Arial' # Для Windows
mpl.rcParams['font.fantasy'] = 'Ubuntu' # Для Ubuntu
mpl.rcParams['font.fantasy'] = 'Arial, Ubuntu'

while True:
    try:
        print('Введите директорию, в которой хранятся тренировочные изображения:')
        print('/home/lena/Desktop/traffic-sign-recognition/Знаки ПДД/Тренировочные данные')
        train_data_dir = '/home/lena/Desktop/traffic-sign-recognition/Знаки ПДД/Тренировочные данные'
        # train_data_dir = askdirectory()
        print('Выбранная директория: ', train_data_dir)
        print('Проводится загрузка тренировочных изображений...')
        start_time = time.time()
        images, labels, labels_str = load_data(train_data_dir)
        break
    except Exception:
        print('Ошибка, проверьте корректность пути')

print('Загрузка изображений была успешно завершена за ', time.time() - start_time, ' милисекунд')
print('Всего загружено ', len(images), ' изображений')

amount_images = len(set(labels))

start_time = time.time()
print('Производится сжатие изображений до 32x32 пикселя...')
images32 = [skimage.transform.resize(image, (32, 32)) for image in images]
print('Сжатие изображений завершено за ', time.time() - start_time, ' милисекунд')

images_a = np.array(images32)
labels_a = np.array(labels)

print('Создание графа...')
graph = tf.Graph()

print('Введите параметры обучения:')
learning_rate = float(input("Скорость обучения (learning rate): "))
# learning_rate = 0.1
learning_steps = int(input("Количество итераций обучения: "))
# learning_steps = 100

with graph.as_default():
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 4])
    labels_ph = tf.placeholder(tf.int32, [None])

    images_flat = tf.contrib.layers.flatten(images_ph)
    logits = tf.contrib.layers.fully_connected(images_flat, amount_images, tf.nn.relu)
    predicted_labels = tf.argmax(logits, 1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph,
                                                                         logits=logits))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
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
    print('Что вы хотите сделать? \n' +
          '1: Проверить правдивость распознавателя на тренировочных данных. \n' +
          '2: Распознать изображения; \n'
          '3: Выход.')
    type = int(input("Введите число: "))
    if type == 1 :
        try:
            test_data(labels_str)
        except Exception:
            print('Ошибка, проверьте корректность пути')
    elif type == 2:
        while True:
            try:
                run_on_image(labels_str)
                plt.ion()
                plt.show()
                ans = input('Продолжить распознавания изображений? y/n: ')
                if ans == 'n':
                    break
            except Exception:
                print('Ошибка, проверьте корректность пути')
    elif type == 3:
        break

session.close()