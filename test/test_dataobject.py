# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

Unittests for dataobject.py
"""

import os
import sys
"""
# Due to geneflow.py is on another level, it is neccesary to indicates parent path
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath("geneflow.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import geneflow as genf

import unittest

# A unit test is a test that checks a single component of code, 
# usually modularized as a function, and ensures that it performs as expected.
# Unit tests in PyUnit are structured as subclasses of the unittest.TestCase class,
# and we can override the runTest() method to perform our own unit tests
# which check conditions using different assert functions in unittest.TestCase
# https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/


# The test based on unittest module
class TestDataObject(unittest.TestCase):
    pass