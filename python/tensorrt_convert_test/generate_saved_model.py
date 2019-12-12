from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import os

### Generate a Train Model

x = tf.placeholder(tf.float32, [None, 784], name="myInput")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="myOutput")

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


### EXPORT_MODEL


### SavedModel using Simple SAVE 
saved_model_dir = "output_savedmodel_dir"
if os.path.exists(saved_model_dir):
    os.remove(saved_model_dir)


tf.saved_model.simple_save(
    sess,    
    saved_model_dir,
    inputs={"myInput":x}, 
    outputs={"myOutput":y}
)


### SavedModel using My Own TAG
saved_model_dir2 = "output_savedmodel_dir2"
if os.path.exists(saved_model_dir2):
    os.remove(saved_model_dir2)
   

builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir2)
signature = predict_signature_def(
    inputs={'myInput': x}, 
    outputs={'myOutput':y}
)

# using custom age instead of default tag
builder.add_meta_graph_and_variables(
    sess=sess,
    tags=["myTag"],
    signature_def_map={'predict':signature}
)

builder.save()





