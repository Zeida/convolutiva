# -*- coding: utf-8 -*-

# Sample code to use string producer.

"""Librerias a usar tensorflow y numpy"""
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plot

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 3

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------


def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()

        _, file_image = reader.read(filename_queue)
        #label = one_hot
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)

        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        #mezcla las imagenes
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)

        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)
    #tensor
    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#               MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    """Estructuras with es una estructura de contexto"""

    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        print (tf.shape(o3))
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        h = tf.layers.dense(inputs=o4, units=30, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
        print (tf.shape(y))
    return y

#rutas de ficheros de entrenamiento validacion y test
example_batch_train, label_batch_train = dataSource(["training/bart_simpson/*.jpg", "training/lisa_simpson/*.jpg",
                                                        "training/maggie_simpson/*.jpg"], batch_size=batch_size)

example_batch_valid, label_batch_valid = dataSource(["valid/bart_simpson/*.jpg", "valid/lisa_simpson/*.jpg",
                                                        "valid/maggie_simpson/*.jpg"], batch_size=batch_size)

example_batch_test, label_batch_test = dataSource(["test/bart_simpson/*.jpg", "test/lisa_simpson/*.jpg",
                                                        "test/maggie_simpson/*.jpg"], batch_size=batch_size)



example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train,tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid,tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_test,tf.float32)))
#cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_train,tf.float32) * tf.log(example_batch_train_predicted), reduction_indices=[1]))
#cost_valid = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_valid,tf.float32) * tf.log(example_batch_valid_predicted), reduction_indices=[1]))
#cost_test = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_test,tf.float32) * tf.log(example_batch_test_predicted), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_train = []
    error_valid = []
    diferencia_error = 100.0

    error_train.append(sess.run(cost))

    print("Iter:", 0, "---------------------------------------------")
    print(sess.run(label_batch_valid))
    print(sess.run(example_batch_valid_predicted))
    error_valid.append(sess.run(cost_valid))
    print("Error:", error_valid[-1])
    iteration = 1


    while diferencia_error > (1/(10**5)):
        sess.run(optimizer)

        print("Iter:", iteration, "---------------------------------------------")
        label_valid = sess.run(label_batch_valid)
        predicted_valid = sess.run(example_batch_valid_predicted)
        # for b, r in zip(label_valid, predicted_valid):
        #     print(b, "--->", r)
        error_valid.append(sess.run(cost_valid))
        error_train.append(sess.run(cost))
        print("Error entrenamiento:", error_train[-1])
        print("Error:", error_valid[-1])
        diferencia_error = abs(error_valid[-2] - error_valid[-1])
        print("Diferencia de error: ", diferencia_error)
        iteration += 1
    total = 0.0
    error = 0.0

    test_data = sess.run(label_batch_test)
    test_hoped = sess.run(example_batch_test_predicted)
    for real_data, hoped in zip(test_data, test_hoped):
        if np.argmax(real_data) != np.argmax(hoped):
            error += 1
        total += 1
    fallo = error / total * 100
    print("El porcentaje de error es: ", fallo, "% y el de exito ", (100 - fallo), "%")

    tr_handle, = plot.plot(error_train)
    vl_handle, = plot.plot(error_valid)
    plot.legend(handles=[tr_handle, vl_handle],
                labels=['Error entrenamiento', 'Error validacion'])
    plot.title("Learning rate = 0.001")
    plot.show()
    plot.savefig('Grafica_entrenamiento.png')

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
