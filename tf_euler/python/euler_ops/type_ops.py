# Copyright 2018 Alibaba Inc. All Rights Conserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import os

import tensorflow as tf

from tf_euler.python.euler_ops import base

get_node_type = base._LIB_OP.get_node_type
