# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import os 

CWD = os.getcwd()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

OUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "output")
DEFAULT_DIR   = os.path.join(ROOT_DIRECTORY, "default")
