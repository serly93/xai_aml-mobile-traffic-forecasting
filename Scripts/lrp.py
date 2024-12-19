"""Class that implements the layer-wise relevance propagation algorithm.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import gen_nn_ops


class RelevancePropagation:
    """Very basic implementation of the layer-wise relevance propagation algorithm.
    """

    def __init__(self, model, conf, ind):
        super(RelevancePropagation, self).__init__()
        self.epsilon = conf["lrp"]["epsilon"]
        self.rule = conf["lrp"]["rule"]
        self.pooling_type = conf["lrp"]["pooling_type"]
        self.grayscale = conf["lrp"]["grayscale"]

        # Load model
        input_shape = (conf["image"]["height"], conf["image"]
                       ["width"], conf["image"]["channels"])
        network = conf["model"]["name"]
        weights = conf["model"]["weights"]

        self.model = model
        # Extract model's weights
        self.weights = {weight.name.split('/')[0]: weight for weight in self.model.trainable_weights
                        if 'bias' not in weight.name}
        # Extract activation layers
        self.activations = [
            layer.output for layer in self.model.layers if 'dropout' not in layer.name]
        self.activations = self.activations[::-1]
        # Extract the model's layers name
        self.layer_names = [
            layer.name for layer in self.model.layers if 'dropout' not in layer.name]
        self.layer_names = self.layer_names[::-1]
        # Build relevance graph
        self.relevance = self.relevance_propagation()
        self.f = K.function(inputs=self.model.input, outputs=self.relevance)

    def clear(self):
        self.weights = None
        self.layer_names = None
        self.activations = None

    def run(self, image):
        """Computes feature relevance maps for a single image.

        Args:
            image: array of shape (W, H, C)

        Returns:
            RGB or grayscale relevance map.

        """

        image = tf.expand_dims(image, axis=4)
        relevance_scores = self.f(inputs=image)
        relevance_scores = self.postprocess(relevance_scores)
        self.clear()
        return relevance_scores

    def relevance_propagation(self):
        """Builds graph for relevance propagation."""
        relevance = self.model.output
        for i, layer_name in enumerate(self.layer_names):
            # print("====================================")
            # print(layer_name)
            if 'prediction' in layer_name:
                relevance = self.relprop_dense(
                    self.activations[i + 1], self.weights[layer_name], relevance)
            elif 'dense' in layer_name:
                relevance = self.relprop_dense(
                    self.activations[i + 1], self.weights[layer_name], relevance)
            elif 'flatten' in layer_name:
                relevance = self.relprop_flatten(
                    self.activations[i + 1], relevance)
            elif 'pool' in layer_name:
                relevance = self.relprop_pool(
                    self.activations[i + 1], relevance)
            elif 'conv' in layer_name:
                relevance = self.relprop_conv(
                    self.activations[i + 1], self.weights[layer_name], relevance, layer_name)
            elif 'dropout' in layer_name:
                pass
            elif 'input' in layer_name:
                pass
            else:
                raise Exception("Error: layer type not recognized.")
        return relevance

    def relprop_dense(self, x, w, r):
        """Implements relevance propagation rules for dense layers.

        Args:
            x: array of activations
            w: array of weights
            r: array of relevance scores

        Returns:
            array of relevance scores of same dimension as a

        """
        if self.rule == "z_plus":
            w_pos = tf.maximum(w, 0.0)
            z = tf.matmul(x, w_pos) + self.epsilon
            s = r / z
            c = tf.matmul(s, tf.transpose(w_pos))
            return c * x
        else:
            raise Exception("Error: rule for dense layer not implemented.")

    def relprop_flatten(self, x, r):
        """Transfers relevance scores coming from dense layers to last feature maps of network.

        Args:
            x: array of activations
            r: array of relevance scores

        Returns:
            array of relevance scores of same dimension as a

        """
        return tf.reshape(r, tf.shape(x))

    def relprop_pool(self, x, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'):
        """Implements relevance propagation through pooling layers.

        Args:
            x: array of activations
            r: array of relevance scores
            ksize: pooling kernel dimensions used during forward path
            strides: step size of pooling kernel used during forward path
            padding: parameter for SAME or VALID padding

        Returns:
            array of relevance scores of same dimensions as a

        """

        if self.pooling_type == "avg":
            z = tf.nn.avg_pool(x, ksize, strides, padding) + self.epsilon
            s = r / z
            c = gen_nn_ops.avg_pool_grad(
                tf.shape(x), s, ksize, strides, padding)
        elif self.pooling_type == "max":
            z = tf.nn.max_pool(x, ksize, strides, padding) + self.epsilon
            s = r / z
            c = gen_nn_ops.max_pool_grad_v2(x, z, s, ksize, strides, padding)
        else:
            raise Exception("Error: no such unpooling operation implemented.")
        return c * x

    def relprop_conv(self, x, w, r, name, strides=(1, 1, 1, 1, 1), padding='SAME'):
        """Implements relevance propagation rules for convolutional layers.

        Args:
            x: array of activations
            w: array of weights
            r: array of relevance scores
            name: current layer name
            strides: step size of filters used during forward path
            padding: parameter for SAME or VALID padding

        Returns:
            array of relevance scores of same dimensions as a

        """
        if name == 'block1_conv1':
            x = tf.ones_like(x)     # only for input

        if self.rule == "z_plus":
            w_pos = tf.maximum(w, 0.0)
            z = tf.nn.conv3d(x, w_pos, strides, padding) + self.epsilon
            s = r / z
            c = tf.raw_ops.Conv3DBackpropInputV2(
                input_sizes=tf.shape(x), filter=w_pos, out_backprop=s, strides=strides, padding=padding)
            return c * x
        else:
            raise Exception(
                "Error: rule for convolutional layer not implemented.")

    @staticmethod
    def rescale(x):
        """Rescales relevance scores of a batch of relevance maps between 0 and 1

        Args:
            x: RGB or grayscale relevance maps with dimensions (N, W, H, C) or (N, W, H), respectively.

        Returns:
            Rescaled relevance maps of same dimensions as input

        """
        x_min = np.min(x)
        x_max = np.max(x)
        return x

    def postprocess(self, x):
        """Postprocesses batch of feature relevance scores (relevance_maps).

        Args:
            x: array with dimension (N, W, H, C)

        Returns:
            x: array with dimensions (N, W, H, C) or (N, W, H) depending on if grayscale or not

        """

        x = self.rescale(x)
        return x
