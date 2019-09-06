import tensorflow as tf
import numpy as np
import os
import shutil


input_array = np.zeros((1, 28, 28, 3))
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='JW_Input')
W = tf.Variable(tf.zeros([28, 28, 3, 10]))
y = tf.matmul(x, W, name='JW_Output')


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(y, feed_dict={x: input_array})

    target_dir = 'savedmodel_test_dir'
    if os.path.isdir(target_dir) is True:
        shutil.rmtree(target_dir)

    # Save a SavedModel
    tf.saved_model.simple_save(sess, 'savedmodel_test_dir', inputs={'Input': x}, outputs={'Output': y})

# Convert to TFLite Model from SavedModel
#convert = tf.lite.TFLiteConverter.from_saved_model(target_dir, input_shapes={"JW_Input": [None, 28, 28, 3]})
convert = tf.lite.TFLiteConverter.from_saved_model(target_dir)
convert.convert()
