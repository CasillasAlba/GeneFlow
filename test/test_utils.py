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
    

    def test_copy_object(self):
        
        data = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13"],
            "TCGA-WT-AB44": [716.0, 2077.0, 1790.0],
            "TCGA-XX-A899": [24.0, 20.0, 1021.0],
            "TCGA-XX-A89A": [617.0, 1588.0, 1216.0]
        }

        df = pd.DataFrame(data)

        df_copy = ut.copy_object(df)

        assert df_copy.equals(df), "is not a copy"
    
    def test_read_file(self):
        
        path_file = "Una_ruta"

        with self.assertRaises(SystemExit) as cm:
            ut.read_file(path_file, sep="\t")

        self.assertEqual(cm.exception.code, 0)
    

    def test_json_from_dict(self):
        
        my_dict = {
            'flower': {
                'name': 'lotus flower', 'species': 'Nelumbo nucifera'
            }
        }

        my_json = """{"flower": {"name": "lotus flower", "species": "Nelumbo nucifera"}}"""
        
        ut.json_from_dict(my_dict)

        assert my_dict, my_json
    

    def test_dict_from_json(self):
        
        my_json = """{"flower": {"name": "lotus flower", "species": "Nelumbo nucifera"}}"""

        with self.assertRaises(SystemExit) as cm:
            ut.dict_from_json(my_json)

        self.assertEqual(cm.exception.code, 0)



# run the test
unittest.main()