# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

Unittests for utils.py
"""

import os
import sys
"""
# Due to geneflow.py is on another level, it is neccesary to indicates parent path
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath("geneflow.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import src.utils as ut

import unittest

# AUX Import for quick creation
import pandas as pd

# A unit test is a test that checks a single component of code, 
# usually modularized as a function, and ensures that it performs as expected.
# Unit tests in PyUnit are structured as subclasses of the unittest.TestCase class,
# and we can override the runTest() method to perform our own unit tests
# which check conditions using different assert functions in unittest.TestCase
# https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/

# The test based on unittest module
class TestUtils(unittest.TestCase):
    
    def test_server_status(self):
        url = "https://api.gdc.cancer.gov/"
        
        # https://stackoverflow.com/questions/15672151/is-it-possible-for-a-unit-test-to-assert-that-a-method-calls-sys-exit
        with self.assertRaises(SystemExit) as cm:
            ut.check_server_status(url)

        self.assertEqual(cm.exception.code, 0)
        
    


# run the test
unittest.main()