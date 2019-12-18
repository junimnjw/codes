import tensorflow as tf

print("tensorflow version:"+tf.__version__)

with tf.Session(graph=tf.Graph()) as sess:
    export_path = "output_saved_model"
    tf.saved_model.loader.load(sess, ["serve"], export_path)

    graph = tf.get_default_graph()
    print(graph.get_operations())   

    
