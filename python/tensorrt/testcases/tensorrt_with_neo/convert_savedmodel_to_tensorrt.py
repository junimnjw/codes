import timeit
import os
import shutil
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


org_savedmodel_dir = os.path.join(os.getcwd(),"savedmodel_dir_latest")
tensorrt_savedmodel_dir = os.path.join(os.getcwd(),"tensorrt_savedmodel_dir")

if not os.path.exists(org_savedmodel_dir):
    raise FileNotFoundError("notfound")

if os.path.exists(tensorrt_savedmodel_dir):
    shutil.rmtree(tensorrt_savedmodel_dir)
    os.mkdir(tensorrt_savedmodel_dir)

converter = trt.TrtGraphConverter(input_saved_model_dir=org_savedmodel_dir, input_saved_model_signature_key="predict_images",\
        #precision_mode=trt.TrtPrecisionMode.INT8,\
        #use_calibration=False\
        )
converter.convert()
converter.save(tensorrt_savedmodel_dir)

