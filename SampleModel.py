import numpy as np
import tensorflow as tf


class SampleModel:
    def __init__(self, func_tf, func, interval, pos, val, theoretical=None):
        self.tf_equation = func_tf
        self.equation = func
        self.interval = interval
        self.T = interval
        self.pos = pos
        self.val = val
        self.theoretical = theoretical
        self.dim = len(val)
        self.orders = []
        for i in pos:
            self.orders.append(len(i))
        self.max_order = np.max(self.orders)
        self.equations_amount = len(pos)

    def get_pos(self, equation_idx):
        return self.pos[equation_idx]
    
    def get_val(self, equation_idx):
        return self.val[equation_idx]

    def get_init_point(self, index):
        return self.init_points[index]
