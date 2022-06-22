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
        self.category_num = self.config.