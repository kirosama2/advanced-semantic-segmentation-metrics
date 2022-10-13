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
                self.net["drop_prob"] = tf.Variable(1.0)

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("vgg") as scope:
            # build block
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
            fc1 = self.build_fc(block,["fc6_1","relu6_1","drop6_1","fc7_1","relu7_1","drop7_1","fc8_1"], dilate_rate=6)
            fc2 = self.build_fc(block,["fc6_2","relu6_2","drop6_2"