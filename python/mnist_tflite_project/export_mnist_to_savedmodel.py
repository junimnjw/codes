import tensorflow as tf
import os
import shutil

##################
# Generate a Model
##################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.compat.v1.placeholder(tf.float32, [None, None], name='jw_input')
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
k = tf.matmul(x, W) + b
y = tf.nn.softmax(k)
y = tf.argmax(y, 1, name='jw_output')
y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=k, labels=y_), name="JW_CrossEntropy")
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name="JW_TrainStep")

with tf.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Creates SavedModelBuilder class
    # Defines where the model will be exported
    export_path_base = "export_dir_for_savedmodel"
    export_path = os.path.join(export_path_base, str(1.0))
    print('Exporting trained model to', export_path)

    if os.path.isdir(export_path):
        shutil.rmtree(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Creates the TensorInfo protobuf objects that encapsulates the input and output tensors
    tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(y)

    # Defines the MNIST signatures, uses the TF Predict API
    # It receives an image and output
    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_input},
            outputs={'candidate_number': tensor_info_output},
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        )
    )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'classify_digit_images': classification_signature
        }
    )

    # Export the model
    builder.save(as_text=True)
    print('Done exporting!')


#########
# TFLite Convert
#########
src_model_file = os.path.join(os.getcwd(), export_path)
convert = tf.lite.TFLiteConverter.from_saved_model(src_model_file,
                                                   input_arrays=['jw_input'],
                                                   output_arrays=['jw_output'],
                                                   input_shapes={"jw_input": [None, 28 * 28]},
                                                   signature_key='classify_digit_images'
                                                   )

tflite_model_dir = os.path.join(os.getcwd(), 'export_dir_for_tflite')
tflite_model_path = os.path.join(tflite_model_dir, "converted_model.tflite")

convert.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

if os.path.isdir(tflite_model_dir) is True:
    shutil.rmtree(tflite_model_dir)

os.mkdir(tflite_model_dir)
tflite_model = convert.convert()
open(tflite_model_path, "wb").write(tflite_model)

print('Done converting!')
