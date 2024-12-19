import pandas as pd
import glob
import os
import numpy as np
import statistics
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import optimizers, activations, initializers, regularizers, constraints
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, InputLayer, BatchNormalization, MaxPooling3D
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from datetime import datetime, timezone,timedelta
from dateutil import tz
import tensorflow.keras.backend as K
from tensorflow import keras
import math
from IPython.display import display, HTML
from IPython.display import HTML
import random

# The following two functions are from: https://github.com/wnlUc3m/deepcog
def evaluate_costs_single_clust(pred_load, real_load, traffic_peak, alpha):
    # Compute the difference between predicted and real load (error)
    error = pred_load - real_load
    # Compute overprovisioning and SLA violations
    tot_overprov = np.sum(error[np.where(error >= 0)])
    num_viol = len(error[np.where(error < 0)])
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    tot_overprov = np.array(tot_overprov, dtype = float)    
    # Compute total cost
    total_cost = sla_viol + tot_overprov
    return num_viol

def cost_func_more_args(alpha):
    def cost_func(y_true, y_pred):
        epsilon = 0.1
        diff = y_pred - y_true
        cost = np.zeros((diff.shape[0], diff.shape[1]))
        y1 = -epsilon * diff + alpha
        y2 = -np.true_divide(1, epsilon) * diff + alpha
        #y3 = np.true_divide(alpha, 1-(epsilon*alpha)) * (diff - (epsilon*alpha)) 
        y3 = -epsilon * alpha + diff 
        cost = tf.where(diff > (epsilon*alpha), y3, cost)
        cost = tf.where(diff < 0, y1, cost)
        cost = tf.where(tf.logical_and((diff <= (epsilon*alpha)), (diff >= 0)), y2, cost)
        cost = K.mean(cost, axis=-1)
        return cost
    return cost_func

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
        #print(roww)
        row2= np.vstack(row1)
        row3=np.append(row3,row2).astype(int)
    return row3

def mae(y_true, y_pred):
    mae = K.mean(K.abs(y_true - y_pred), axis = -1)
    return mae

# The following three functions are taken from https://github.com/dependable-cps/adversarial-MTSR
def compute_gradient(model_fn, loss_fn, x, y, targeted):
    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad

def fgsm(X, Y, model, loss_fn , epsilon, targeted = False):
    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model,loss_fn, ten_X, Y, targeted)
    dir = np.sign(grad)
    x = np.where(dir == -1)
    dir[x] = 0
    return X + epsilon * dir, Y

def bim(X, Y, model, loss_fn, epsilon, alpha2, I, targeted= False):
    Xp = np.zeros_like(X)
    for t in range(I):
        ten_X = tf.convert_to_tensor(X)
        grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
        dir = np.sign(grad)
        x = np.where(dir == -1)
        dir[x] = 0
        Xp = Xp + (alpha2 * dir)
        Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
        Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
    return Xp, Y

def sla_num(pred_load, real_load):
    error = np.array(pred_load) - np.array(real_load)
    num_viol = np.count_nonzero(error < 0)
    return num_viol

def sla_cost(pred_load, real_load, traffic_peak, alpha):
    error = pred_load - real_load
    num_viol = np.count_nonzero(error < 0)
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)  
    return sla_viol

def overprov_cost(pred_load, real_load, traffic_peak, alpha):
    error = pred_load - real_load
    tot_overprov = np.sum(np.count_nonzero(error >= 0))
    num_viol = np.count_nonzero(error < 0)
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    tot_overprov = np.array(tot_overprov, dtype = float)    
    return tot_overprov

def total_cost(pred_load, real_load, traffic_peak, alpha):
    error = pred_load - real_load
    tot_overprov = np.sum(error[np.where(error >= 0)])
    num_viol = np.count_nonzero(error < 0)
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    tot_overprov = np.array(tot_overprov, dtype = float)    
    total_cost = sla_viol + tot_overprov
    return total_cost

