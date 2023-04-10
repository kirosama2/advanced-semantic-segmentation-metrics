
import os
import re
import sys
import glob
import json
import time
import numpy as np 
import skimage
import skimage.io as imgio
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def crf_inference(img,crf_config,category_num,feat=None,pred=None,gt_prob=0.7,use_log=False):