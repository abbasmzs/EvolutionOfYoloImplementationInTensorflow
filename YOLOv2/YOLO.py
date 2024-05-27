import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np
import keras as K
import cv2 as cv
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import glob
import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """
    box_scores = box_confidence * box_class_probs  # 19*19*5*80
    box_classes = tf.math.argmax(box_scores, axis=-1)  # 19*19*5
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1, keepdims=False)  # 19*19*5
    filtering_mask = box_class_scores >= threshold  # 19*19*5
    scores = tf.boolean_mask(box_class_scores, filtering_mask)  # (? ,)
    boxes = tf.boolean_mask(boxes, filtering_mask)  # (? , 4)
    classes = tf.boolean_mask(box_classes, filtering_mask)  # (? ,)
    return scores, boxes, classes
def UseModel(image_shape):
    # sess = K.get_session()
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    yolo_model = yolo_body(K.Input(image_shape), len(anchors), len(class_names))
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    print(yolo_outputs)
    model = keras.Model(yolo_model.input, yolo_outputs)
    model.summary()
    return model
image_shape = (720, 1280, 3)
input_images = []
for i in glob.glob("image/*"):
    if ".jpg" in i:
        image = cv.resize(cv.imread(i, -1), (image_shape[1], image_shape[0]))
        input_images.append(image)
        print(input_images[-1].shape)
# yolo_outputs = yolo_head(label_output, anchors, len(class_names))
# yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6)
model = UseModel(image_shape)
# model.save("YoloVersion-2")
label_output = model.predict(np.array(input_images))
# print(label_output)
# box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
# converter = trt.TrtGraphConverterV2(
#    input_saved_model_dir="YoloVersion-2",
#    precision_mode=trt.TrtPrecisionMode.FP32
# )
 
# # Convert the model into TRT compatible segments
# trt_func = converter.convert()
# converter.summary()