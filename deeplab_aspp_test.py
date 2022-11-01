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
            fc2 = self.build_fc(block,["fc6_2","relu6_2","drop6_2","fc7_2","relu7_2","drop7_2","fc8_2"], dilate_rate=12)
            fc3 = self.build_fc(block,["fc6_3","relu6_3","drop6_3","fc7_3","relu7_3","drop7_3","fc8_3"], dilate_rate=18)
            fc4 = self.build_fc(block,["fc6_4","relu6_4","drop6_4","fc7_4","relu7_4","drop7_4","fc8_4"], dilate_rate=24)
            #self.net["fc8"] = (self.net[fc1]+self.net[fc2]+self.net[fc3]+self.net[fc4])/4.0
            self.net["fc8"] = self.net[fc1]+self.net[fc2]+self.net[fc3]+self.net[fc4]

            return self.net["fc8"] # note that the output is a log-number

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scop