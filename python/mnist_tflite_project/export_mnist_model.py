import tensorflow as tf
import os
import shutil
import numpy as np
from time import process_time

##################
# Generate a Model
##################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.compat.v1.placeholder(tf.float32, [None, None], name="JW_Input")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
k = tf.matmul(x, W) + b
y = tf.nn.softmax(k, name="JW_Output")
y = tf.argmax(y, 1)
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
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(1.0)))
    print('Exporting trained model to', export_path)
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

    if os.path.isdir(export_path):
        shutil.rmtree(export_path)

    # Export the model
    builder.save(as_text=True)
    print('Done exporting!')

    ###########
    # # Save a Model as savedmodel
    ###########
    # export_dir_for_savedmodel = "output_dir_for_savedmodel"
    # if os.path.isdir(export_dir_for_savedmodel) is True:
    #     shutil.rmtree(export_dir_for_savedmodel)
    #
    # tf.compat.v1.saved_model.simple_save(session=sess,
    #                                      export_dir=export_dir_for_savedmodel,
    #                                      inputs={'Input': x},
    #                                      outputs={'Output': y})

    ##########
    # Evaluation
    ##########

    correct_prediction = tf.equal(y, tf.math.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("tensorflow org evaluation:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#########
# TFLite Convert
#########
isSavedModel = True

if isSavedModel is True:
    src_model_file = os.path.join(os.getcwd(), "output_dir_for_savedmodel")
    convert = tf.lite.TFLiteConverter.from_saved_model(src_model_file,
                                                       input_arrays=["JW_Input"],
                                                       output_arrays=["JW_Output"],
                                                       input_shapes={"JW_Input": [None, 28 * 28]}
                                                       )
else:
    src_model_file = os.path.join(os.getcwd(), "output_dir_for_saver_model", "frozen_graph.pb")
    convert = tf.lite.TFLiteConverter.from_frozen_graph(src_model_file,
                                                        input_arrays=["JW_Input"],
                                                        output_arrays=["JW_Output"])

tflite_model_dir = os.path.join(os.getcwd(), "output_dir_for_tflite")
tflite_model_path = os.path.join(tflite_model_dir, "converted_model.tflite")

convert.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

if os.path.isdir(tflite_model_dir) is True:
    shutil.rmtree(tflite_model_dir)

os.mkdir(tflite_model_dir)
tflite_model = convert.convert()
open(tflite_model_path, "wb").write(tflite_model)

#########
# TFLite Load
#########

with tf.Session() as sess:

    tflite_model_path = os.path.join(os.getcwd(), "output_dir_for_tflite", "converted_model.tflite")
    if not os.path.exists(tflite_model_path) is True:
        raise FileNotFoundError

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Single Image Test
    # start_time = process_time()
    # input_data = np.array(mnist.test.images[0], dtype=np.float32)
    # input_data.shape = (1, 784)
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()
    # ouput_data = interpreter.get_tensor(output_details[0]['index'])
    # if np.argmax(mnist.test.labels[0]) == np.argmax(ouput_data):
    #     print("corrent")
    # elapsed_time = process_time() - start_time
    # print("fps:%.8f" % elapsed_time)

    start_time = process_time()
    num_correct = 0
    for i in range(mnist.test.num_examples):
        input_data = np.array(mnist.test.images[i], dtype=np.float32)
        input_data.shape = (1, 784)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(mnist.test.labels[i]) == np.argmax(output_data):
            num_correct += 1

    print("tensorflow lite evaluation:", num_correct / mnist.test.num_examples)
    elapsed_time = process_time() - start_time
    print("elapsed_time:", elapsed_time)
    #print("fps:%.8f" % (1000 * mnist.test.num_examples / elapsed_time))





