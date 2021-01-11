import requests
import imutils
import os
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime



class DeepLabModel(object):

    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):

        """Creates and loads pretrained deeplab model."""

        self.graph = tf.Graph()

        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):

        """Runs inference on a single image.
            Args:
            image: A PIL.Image object, raw input image.
            Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """

        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map

modelType = "body_parsing/models/xception_model"

MODEL = DeepLabModel(modelType)
def parse(img):
    # Inferences DeepLab model and visualizes result.
    original_im = Image.fromarray(img)

    resized_im, seg_map = MODEL.run(original_im)
    seg_map = cv2.resize(seg_map.astype(float), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return seg_map
