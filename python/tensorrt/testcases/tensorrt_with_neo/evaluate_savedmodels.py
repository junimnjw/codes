import time
import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

test_input_dir = os.path.join(os.getcwd(), 'test-input')
test_output_dir = os.path.join(os.getcwd(),'test-output')

org_savedmodel_dir = os.path.join(os.getcwd(),"savedmodel_dir_old")
tensorrt_savedmodel_dir = os.path.join(os.getcwd(),"tensorrt_savedmodel_dir")

img_path = os.path.join(test_input_dir, "resized_deepfake_0.PNG")
if not os.path.exists(img_path):
        raise FileNotFoundError("NotFound")

if not os.path.exists(org_savedmodel_dir):
    raise FileNotFoundError("notfound")

if not os.path.exists(org_savedmodel_dir):
    raise FileNotFoundError("NotFound")

if not os.path.exists(tensorrt_savedmodel_dir):
    raise FileNotFoundError("NotFound")

img_src = np.array(Image.open(img_path))        
height = img_src.shape[0]
width = img_src.shape[1]
padded_image = np.zeros((513, 513, 3), dtype=np.float32)                
padded_image[0:width, 0:height, 0:2] = img_src[0:width, 0:height, 0:2]

img_path = os.path.join(test_input_dir, "resized_deepfake_0.PNG")
if not os.path.exists(img_path):
    raise FileNotFoundError("NotFound")

# Evaluation for Original SavedModel

with tf.compat.v1.Session() as sess:

    meta_graph = tf.compat.v1.saved_model.load(sess, {'serve'}, org_savedmodel_dir)
    model_signature = meta_graph.signature_def['predict_images']
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs

    feed_dict = {sess.graph.get_tensor_by_name("Placeholder:0"):padded_image, \
            sess.graph.get_tensor_by_name("height:0"):513,\
            sess.graph.get_tensor_by_name("width:0"):513}
    output = sess.graph.get_tensor_by_name("probmap:0")
    start_time = time.time()
    results = sess.run(output, feed_dict = feed_dict)

    print("original savedmodel elapsed time:", time.time() - start_time)


tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:

    meta_graph = tf.compat.v1.saved_model.load(sess, {'serve'}, tensorrt_savedmodel_dir)
    model_signature = meta_graph.signature_def['predict_images']
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs

    feed_dict = {sess.graph.get_tensor_by_name("Placeholder:0"):padded_image, \
            sess.graph.get_tensor_by_name("height:0"):513,\
            sess.graph.get_tensor_by_name("width:0"):513}
    output = sess.graph.get_tensor_by_name("probmap:0")
    start_time = time.time()
    results = sess.run(output, feed_dict = feed_dict)

    print("tensorrt savedmodel elapsed time:", time.time() - start_time)

