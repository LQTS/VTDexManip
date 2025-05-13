import json
from scipy.spatial.transform import Rotation as R
import numpy as np
class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def pretty(d):
    return json.dumps(d, indent=4, ensure_ascii=False)

def get_T_matrix(Roation, xyz):
    if Roation.size == 3:
        matrix = R.from_euler('xyz', Roation.flatten(), degrees=False).as_matrix()
    elif Roation.size == 9:
        matrix = Roation
    else:
        raise AttributeError
    T = np.eye(4)
    T[:3, :3] = matrix
    T[:3, -1] = xyz

    return T
def get_quat(matrix):
    quat = R.from_matrix(matrix).as_quat()
    return quat

import torch
import matplotlib.pyplot as plt

def plot_tensor_data(y: list, title:str, x_name: str = "step", y_name: str="value"):
    # Extract x and y coordinates
    x = np.arange(len(y))

    fig, ax = plt.subplots()
    # Create a scatter plot
    ax.scatter(x, y, label='Original Data')

    # Set labels and title
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    ax.set_ylim(0,15)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()

