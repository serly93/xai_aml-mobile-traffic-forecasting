"""Helper functions for relevance propagation."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
def plot_relevance_map(image, relevance_map, norm=True):
    """Plots original image next to corresponding relevance map.

    Args:
        image: original image
        relevance_map: relevance map of original image
        res_dir: path to directory where results are stored
        i: counter
    """

    if norm == True:
        relevance_map = relevance_map / np.max(relevance_map)
    image = relevance_map
    image2 = image
    image = np.mean(image, axis=0)
    lookback, dim1, dim2 = np.shape(image)

    def f(relevance_map, size, position):
        size = np.int_(size)
        position = np.int_(position)
        lrp = relevance_map[position:position + size]
        return np.mean(lrp, axis=0)

    col = lookback
    row = np.floor((lookback - 1) / col) + 1

    fig, axs = plt.subplots(nrows=int(row), ncols=int(col))
    imt_l = []
    for ind, ax in enumerate(axs.flat):
        imt = ax.pcolormesh(image[ind], cmap='viridis',
                            edgecolor='k', lw=2, vmin=0, vmax= 1)
        imt_l.append(imt)
        ax.set_xlabel("Time" + str(-3 + ind))
        ax.set_ylim(ax.get_ylim()[::-1])
    fig.subplots_adjust(right=0.80)
    fig.subplots_adjust(bottom=0.30)
    cbar_ax = fig.add_axes([0.85, 0.30, 0.05, 0.65])
    fig.colorbar(imt, cax=cbar_ax)

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    size_slider = Slider(
        ax=axfreq,
        label='Moving average size',
        valmin=1,
        valmax=100,
        valfmt='%0.0f',
        valinit=1,
    )
    axgate = plt.axes([0.25, 0.15, 0.65, 0.03])
    gate_slider = Slider(
        ax=axgate,
        label='Gate Slide',
        valmin=0,
        valmax=455,
        valfmt='%0.0f',
        valinit=0,
    )

    def update(val):
        relev = f(relevance_map, size_slider.val, gate_slider.val)
        for ind, imt in enumerate(imt_l):
            imt.set_array(relev[ind])
        fig.canvas.draw()

    size_slider.on_changed(update)
    gate_slider.on_changed(update)

    plt.show()