
import os
import sys
import math
import numpy as np
import tensorflow as tf
import skimage.color as imgco

class dataset():
    def __init__(self,config={}):
        self.config = config
        if self.config["input_size"] is not None:
            self.w,self.h = self.config.get("input_size",(240,240))
        else:
            self.w,self.h = None,None
        self.categorys = self.config.get("categorys",["train","val"])
        assert len(self.categorys) > 0, "no enough categorys in dataset"
        self.main_path = self.config.get("main_path",os.path.join("pascal","VOCdevkit","VOC2012"))
        #self.main_path = self.config.get("main_path",os.path.join("pascal"))
        self.ignore_label = self.config.get("ignore_label",255)
        self.category_num = self.config.get("category_num",21)
        self.default_category = self.config.get("default_category",self.categorys[0])
        self.img_mean = np.ones((1,1,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434

        self.data_f,self.data_len = self.get_data_f()

    def get_data_f(self):
        data_f = {}
        data_len = {}
        for category in self.categorys:
            data_f[category] = {"img":[],"label":[],"id":[]}
            data_len[category] = 0
        for one in self.categorys:
            with open(os.path.join("pascal","txt","%s.txt" % one),"r") as f:
                for line in f.readlines():
                    line = line.strip("\n") # the line is like "2007_000738"
                    #line = "2007_000332"
                    data_f[one]["id"].append(line)
                    data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % line))
                    data_f[one]["label"].append(os.path.join(self.main_path,"SegmentationClassAug","%s.png" % line))
                if "length" in self.config:
                    length = self.config["length"]
                    data_f[one]["id"] = data_f[one]["id"][:length]
                    data_f[one]["img"] = data_f[one]["img"][:length]
                    data_f[one]["label"] = data_f[one]["label"][:length]
            data_len[one] = len(data_f[one]["label"])

        print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,batch_size=None,epoches=-1):
        if category is None: category = self.default_category
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
        if self.h is None: assert batch_size == 1,"the input size is None, so the batch size must be 1"
        dataset = tf.data.Dataset.from_tensor_slices({
            "id":self.data_f[category]["id"],
            "img_f":self.data_f[category]["img"],
            "label_f":self.data_f[category]["label"]
            })
        def m(x):
            id = x["id"]
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img = tf.expand_dims(img,axis=0)
            label_f = x["label_f"]
            label_raw = tf.read_file(label_f)
            label = tf.image.decode_image(label_raw)[:,:,0:1]