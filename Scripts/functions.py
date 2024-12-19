import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras import Model
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from lrp import RelevancePropagation
from tensorflow.python.framework.ops import (disable_eager_execution)
from utils_lrp import plot_relevance_map
import math
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import shutil

def cost_func_more_args(alpha):
    # https://github.com/wnlUc3m/deepcog
    def cost_func(y_true, y_pred):
        alpha = 0.1
        epsilon = 0.1
        diff = y_pred - y_true
        cost = np.zeros(diff.shape[0])
        y1 = -epsilon * diff + alpha
        y2 = -np.true_divide(1, epsilon) * diff + alpha
        # y3 = np.true_divide(alpha, 1-(epsilon*alpha)) * (diff - (epsilon*alpha))
        y3 = -epsilon * alpha + diff
        cost = tf.where(diff > (epsilon * alpha), y3, cost)
        cost = tf.where(diff < 0, y1, cost)
        cost = tf.where(tf.logical_and(
            (diff <= (epsilon * alpha)), (diff >= 0)), y2, cost)
        cost = K.mean(cost, axis=-1)
        return cost
    return cost_func

def prep_data(path, path2):
    input = np.load(path)
    nn = tf.keras.models.load_model(
        path2, custom_objects={'loss': cost_func_more_args}, compile=False)
    return input, nn

def extract_windows(array, clearing_time_index, max_time, sub_window_size):
    examples = []
    start = clearing_time_index + 1 - sub_window_size + 1

    for i in range(max_time + 1):
        example = array[start + i:start + sub_window_size + i] 
        examples.append(np.expand_dims(example, 0))
    return np.vstack(examples)

def layer_wise_relevance_propagation(model, conf, image, ind):

    img_dir = conf["paths"]["image_dir"]
    res_dir = conf["paths"]["results_dir"]

    image_height = conf["image"]["height"]
    image_width = conf["image"]["width"]

    lrp = RelevancePropagation(model, conf, ind)
    relevance_map = lrp.run(image)
    return relevance_map

def moving_average(final_image, samples):
    dims = list(np.shape(final_image))
    dims[0] -= samples - 1
    new_tensor = np.zeros(dims)
    for ind in range(dims[0]):
        new_tensor[ind, :] = np.mean(final_image[ind:ind + samples], axis = 0)
    return new_tensor

# Generate neighboring cell ids for Milan dataset
def get_rows(cell_id, nr):
    row2 = []
    row3 = []
    for i in range(1,nr+1):
        if i> math.ceil(nr/2):
            globals()["row_%d" %i] = np.arange(100 * (i-math.ceil(nr/2)) + cell_id - math.floor(nr/2),100 * (i-math.ceil(nr/2)) + cell_id + math.floor(nr/2)+1)
        elif i<math.ceil(nr/2):
             globals()["row_%d" %i] = np.arange(cell_id - math.floor(nr/2) - 100 * (math.ceil(nr/2)-i) , cell_id + math.floor(nr/2) + 1 - 100 * (math.ceil(nr/2)-i))
        else:
             globals()["row_%d" %(math.ceil(nr/2))] = np.arange(cell_id - math.floor(nr/2), cell_id + math.floor(nr/2)+1)

    for j in range(1,nr+1):
        row1=globals()["row_%d" %j]
        row2= np.vstack(row1)
        row3=np.append(row3,row2).astype(int)
    return row3
