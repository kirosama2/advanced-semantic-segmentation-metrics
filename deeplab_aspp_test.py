import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from pythonlib.crf import crf_inference
from pythonlib.dataset import dataset
from pythonlib.predict import Predict

class Test():
    def __init__(self,config):
        self.config = config
        if self.config["input_size"] is not None:
            self.h,self.w = self.config.get("input_size",(25,25))
        else:
            self.h,self.w = None,None
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.net = {}
        self.weights = {}
        self.min_prob = 0.0001
        self.stride = {}
        self.stride["input"] = 1
        self.trainable_list = []

    def build(self,net_input,net_label,net_id):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = net_input
                self.net["label"] = net_label # [None, self.h,self.w,1], int32
                self.net["id"] = net_id
                self.net["drop_prob"] = tf.Variable