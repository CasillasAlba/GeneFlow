# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

Unittests for processing.py
"""

import os
import sys
"""
# Due to geneflow.py is on another level, it is neccesary to indicates parent path
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath("geneflow.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest

# AUX Import for quick creation
import pandas as pd
import src.processing as pro
import src.utils as ut


# A unit test is a test that checks a single component of code, 
# usually modularized as a function, and ensures that it performs as expected.
# Unit tests in PyUnit are structured as subclasses of the unittest.TestCase class,
# and we can override the runTest() method to perform our own unit tests
# which check conditions using different assert functions in unittest.TestCase
# https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/

# The test based on unittest module
class TestAnalysis(unittest.TestCase):

    def test_counts(self):
        
        data = {
            "Sport" : ["Footbal", "Basketball", "Basketball", "Tennis"],
            "calories": [420, 380, 390, 380]
        }
        
        df = pd.DataFrame(data)
        
        assert pro.CountTypes("Sports").apply(df) == -1, "ERROR"
        
    
    def test_duplicates_columns(self):
        # Datas with Duplicates - Rows
        data_dup_columns = [
            ["Footbal",     420, 50, 0, 120, 0],
            ["Basketball",  380, 40, 3, 840, 2],
            ["Tennis",      390, 45, 3, 345, 7],
        ]
        
        columns_df = ["Sport", "calories", "duration", "days", "temp", "days"]

        # We can't create a dataframe with duplicated columns but we can join it!!
        data_dup = pd.DataFrame(data_dup_columns, columns = columns_df)
        
        data_dup = data_dup.set_index("Sport")
        
        result = [
            ["Footbal",     0, 0],
            ["Basketball",  3, 2],
            ["Tennis",      3, 7],
        ]
        
        columns_dup = ["Sport", "days", "days"]
        
        result_dup = pd.DataFrame(result, columns=columns_dup)
        
        result_dup = result_dup.set_index("Sport")
        
        assert pro.DataDuplicates(axis = 1).apply(data_dup).equals(result_dup), "ERROR"
        
    
    def test_duplicates_row(self):
        # Datas with Duplicates - Rows
        data_dup_rows = {
            "Sport" : ["Footbal", "Basketball", "Basketball", "Tennis"],
            "calories": [420, 380, 390, 380],
            "duration": [50, 40, 45, 40],
            "temp": [120, 840, 345, 840],
            "days": [0, 3, 5, 3]
        }
        
        data_dup = pd.DataFrame(data_dup_rows)
        
        data_dup = data_dup.set_index("Sport")
        
        result = {
            "Sport" : ["Basketball", "Basketball"],
            "calories": [380, 390],
            "duration": [40, 45],
            "temp": [840, 345],
            "days": [3, 5]
        }
        
        result_dup = pd.DataFrame(result)
        
        result_dup = result_dup.set_index("Sport")
        
        assert pro.DataDuplicates(axis = 0).apply(data_dup).equals(result_dup), "ERROR"

    
    def test_replace_value(self):
        data = {
          "calories": [420, "Unknown", 390],
          "duration": [50, 40, "Unknown"]
        }
        
        df = pd.DataFrame(data)
        
        result = {
          "calories": [420, 0, 390],
          "duration": [50, 40, 0]
        }
        
        res = pd.DataFrame(result)

        assert pro.Replace(replaced_by = 0).apply(df).equals(res), "ERROR"
        
    
    def test_replace_value2(self):
        data = {
          "calories": [420, "Not Available", 390],
          "duration": [50, 40, "Not Available"]
        }
        
        df = pd.DataFrame(data)
        
        result = {
          "calories": [420, "Not Available", 390],
          "duration": [50, 40, "Not Available"]
        }
        
        res = pd.DataFrame(result)

        assert pro.Replace(to_replace = "Not Applicable", replaced_by = 0).apply(df).equals(res), "ERROR"

    
    def test_replace_value3(self):
        data = {
          "calories": [420, 2000, 390],
          "duration": [50, 40, 2000]
        }
        
        df = pd.DataFrame(data)
        
        result = {
          "calories": [420, 2, 390],
          "duration": [50, 40, 2]
        }
        
        res = pd.DataFrame(result)

        assert pro.Replace(to_replace = 2000, replaced_by = 2).apply(df).equals(res), "ERROR"
        
    

    def test_clinical_name(self):
        clinic = "notfound"
        clinical_options = ["drug", "patient", "nte", "radiation", "follow_up", "omf"]
        

        result = pro.CheckElement(clinic).apply(clinical_options)
        
        assert result == False

    
    def test_columnnames(self):
        data = 23

        with self.assertRaises(SystemExit) as cm:
            pro.DataColumnames().apply(data)

        self.assertEqual(cm.exception.code, 0)
    
    
    def test_columnames_correct(self):
        data = {
          "calories": [420, 380, 390],
          "duration": [50, 40, 45]
        }
        
        df = pd.DataFrame(data)
        
        pro.DataColumnames().apply(df) 
        
        columnames = df.columns.values
        
        assert ['calories', 'duration'], columnames


    def test_subdataframe_col(self):
        data = {
          "calories": [420, 380, 390],
          "duration": [50, 40, 45]
        }
        
        df = pd.DataFrame(data)
        
        test_list = ['calories', 'duration', 'speed']
        
        res = pro.DataProjectionList(test_list).apply(df)
        
        assert res.equals(df)

    
    def test_subdataframe_col_2(self):
        data = {
          "calories": [420, 380, 390],
          "duration": [50, 40, 45]
        }
        
        df = pd.DataFrame(data)
        
        data2 = {
          "calories": [420, 380, 390]
        }
        
        df_res = pd.DataFrame(data2)
        
        test_list = ['calories', 'speed']
        
        res = pro.DataProjectionList(test_list).apply(df)

        assert res.equals(df_res)

        
    def test_subdataframe_col_3(self):
        data = {
            "Sport" : ["Footbal", "Basketball", "Tennis"],
            "calories": [420, 380, 390],
            "duration": [50, 40, 45],
            "temp": [120, 840, 345]
        }
        
        df = pd.DataFrame(data)
        
        df = df.set_index("Sport")
        
        test_list = ["calories"]
        
        result = {
            "Sport" : ["Footbal", "Basketball", "Tennis"],
            "calories": [420, 380, 390]
        }
        
        
        res = pd.DataFrame(result)
        res = res.set_index("Sport")
        
        assert pro.DataProjectionList(test_list).apply(df).equals(res), "incorrect columnnames"
        
        
    def test_subdataframe_row(self):
        data = {
            "Sport" : ["Footbal", "Basketball", "Tennis"],
            "calories": [420, 380, 390],
            "duration": [50, 40, 45],
            "temp": [120, 840, 345]
        }
        
        df = pd.DataFrame(data)
        
        df = df.set_index("Sport")
        
        test_list = ["Footbal", "Tennis"]
        
        result = {
            "Sport" : ["Footbal", "Tennis"],
            "calories": [420, 390],
            "duration": [50, 45],
            "temp": [120, 345]
        }

        res = pd.DataFrame(result)
        res = res.set_index("Sport")

        assert pro.DataSelectionList(test_list).apply(df).equals(res), "incorrect rownames"

    
    def test_intersection(self):
        l1 = ['Hi' ,'hello', 'at', 'this', 'there', 'from']
        l2 = ['there' , 'hello', 'hola']
        
        assert pro.ListIntersection(l2).apply(l1) == ['there', 'hello'], "incorrect common elements"
    

    def test_rename_columns(self):
        data = {
          "calories": [420, 380, 390],
          "duration": [50, 40, 45]
        }
        
        df = pd.DataFrame(data)
        
        with self.assertRaises(SystemExit) as cm:
            pro.RenameColumname(-3, 2).apply(df)

        self.assertEqual(cm.exception.code, 0)
        
    def test_rename_columns_2(self):
        data = {
          "calories": [420, 380, 390],
          "duration": [50, 40, 45]
        }
        
        df = pd.DataFrame(data)
        
        with self.assertRaises(SystemExit) as cm:
            pro.RenameColumname(2, 50).apply(df)

        self.assertEqual(cm.exception.code, 0)
    

    def test_metadata(self):
        
        # Custom Class
        class CheckElement():

            def __init__(self, element = ""):

                arguments = locals()
                
                inputs_params = self.get_metadata()["Input Parameters"]

                ut.check_arguments(arguments, inputs_params)
                
                super().__init__()

                self.element = element

            # Parameters

            def get_parameters(self) -> dict:
                parameters = {}
                parameters['element'] = self.element
                return parameters

            def set_parameters(self, parameters: dict):
                self.element = parameters['element']

            # Getters

            def get_element(self):
                return self.element

            def get_metadata(self):
                return {
                    "Input Parameters" : {
                        "element" : {
                            "type" : str,
                            "description" : "Element to find"
                        }
                    },
                    "Apply Input" : {
                        "list_" : {
                            "type" : list,
                            "description" : "The main list to check if element exists"
                        }
                    },
                    "Output" : {
                        "return" : {
                            "type" : bool,
                            "description" : "True if element is on list, False if not on list"
                        }
                    }
                }

            def show_metadata(self):
                ut.print_json(self.get_metadata())

            # Execution
            # Check if a value is in the list
            def apply(self, list_):
                
                arguments = locals()
                
                inputs_params = self.get_metadata()["Apply Input"]

                ut.check_arguments(arguments, inputs_params)
                
                try:
                    if (self.get_element()).lower() not in list_:
                        return False
            
                    return True
                
                except:
                    print("Element have not the same type of list elements")
                    sys.exit(0)


        elm = "Good Morning"

        with self.assertRaises(SystemExit) as cm:
            
            CheckElement(elm).apply(elm)

        self.assertEqual(cm.exception.code, 0)


# run the test
unittest.main()