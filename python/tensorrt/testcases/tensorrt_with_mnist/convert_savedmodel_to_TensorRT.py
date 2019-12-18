from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
images, labels = mnist.test.images, mnist.test.labels

import timeit
import os
import shutil
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

org_savedmodel_dir = "output_savedmodel_dir"
tensorrt_savedmodel_dir = "converted_savedmodel_dir"

if not os.path.exists(org_savedmodel_dir):
	raise FileNotFoundError("notfound")

if os.path.exists(tensorrt_savedmodel_dir):
	shutil.rmtree(tensorrt_savedmodel_dir)
	os.mkdir(tensorrt_savedmodel_dir)

converter = trt.TrtGraphConverter(input_saved_model_dir=org_savedmodel_dir)
converter.convert()
converter.save(tensorrt_savedmodel_dir)


# Evaluation for Original SavedModel
###
with tf.Session() as sess:
	meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], org_savedmodel_dir)
	model_signature = meta_graph.signature_def['serving_default']
	input_signature = model_signature.inputs
	output_signature = model_signature.outputs
	start = timeit.default_timer()
	feed_dict = {sess.graph.get_tensor_by_name(input_signature['myInput'].name):mnist.test.images[:10]}
	output = sess.graph.get_tensor_by_name(output_signature['myOutput'].name)
	results = sess.run(output, feed_dict=feed_dict)
	print("org savedmodel time:", timeit.default_timer() - start)

tf.reset_default_graph()

with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], tensorrt_savedmodel_dir)
    model_signature = meta_graph.signature_def['serving_default']
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs
    start = timeit.default_timer()
    feed_dict = {sess.graph.get_tensor_by_name(input_signature['myInput'].name):mnist.test.images[:10]}
    output = sess.graph.get_tensor_by_name(output_signature['myOutput'].name)
    results = sess.run(output, feed_dict=feed_dict)
    print("tensorrt savedmodel time:", timeit.default_timer() - start)


