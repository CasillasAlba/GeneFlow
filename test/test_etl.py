# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

Unittests for etl.py
"""

import os
import sys
"""
# Due to geneflow.py is on another level, it is neccesary to indicates parent path
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath("geneflow.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.etl import ProcessGDC
from src.objects import GDCQuery

import unittest

# A unit test is a test that checks a single component of code, 
# usually modularized as a function, and ensures that it performs as expected.
# Unit tests in PyUnit are structured as subclasses of the unittest.TestCase class,
# and we can override the runTest() method to perform our own unit tests
# which check conditions using different assert functions in unittest.TestCase
# https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/

# The test based on unittest module
class TestProcessGDC(unittest.TestCase):
    
    def test_search_project(self):
        ident = "TCGA-SKCM"
        
        assert ProcessGDC.search_project(ident) == "TCGA-SKCM", "ERROR"
        
     
    def test_search_project1(self):
        ident = "TCGA-SKC"
        
        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.search_project(ident)
    
        self.assertEqual(cm.exception.code, 0)

        
    def test_search_project2(self):
        ident = "Adrenocortical Carcinoma"
        
        assert ProcessGDC.search_project(ident) == "TCGA-ACC", "ERROR"
        
                
    def test_search_project3(self):
        ident = "AdrenocoRtical CarcinOma"
        
        assert ProcessGDC.search_project(ident) == "TCGA-ACC", "ERROR"
       
                
    def test_search_project4(self):
        ident = "Adrenocortical"
        
        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.search_project(ident)
    
        self.assertEqual(cm.exception.code, 0)
        
       
    def test_regex_filename(self):
        proj_id = "skcm"
        clin_inf = "drug"
        file_name = "nationwidechildrens.org_clinical_patient_skcm.txt"
    
        assert ProcessGDC.regex_file_name(proj_id, clin_inf, file_name) == False, "ERROR"


    def test_regex_filename2(self):
        proj_id = "skcm"
        clin_inf = "patient"
        file_name = "nationwidechildrens.org_clinical_patient_skcm.txt"
    
        assert ProcessGDC.regex_file_name(proj_id, clin_inf, file_name) == True, "ERROR"


    def test_regex_filename3(self):
        proj_id = "skcm"
        clin_inf = "patient"
        file_name = "nationwidechildrens.org_clinical_patient_v1.0_skcm.txt"
    
        assert ProcessGDC.regex_file_name(proj_id, clin_inf, file_name) == True, "ERROR"


    def test_regex_filename4(self):
        proj_id = "skcm"
        clin_inf = "patient"
        file_name = "nationwidechildrens.org_clinical_patient_v0.8_skcm.txt"
    
        assert ProcessGDC.regex_file_name(proj_id, clin_inf, file_name) == True, "ERROR"


    def test_find_clinical(self):
        file_list = ["nationwidechildrens.org_clinical_patient_v1.2_skcm.txt", 
                     "nationwidechildrens.org_clinical_drug_skcm.txt",
                     "nationwidechildrens.org_clinical_treatment_skcm.txt"]
        
        clinical_option = "nationwidechildrens.org_clinical_patient_v1.2_skcm.txt"
    
        assert ProcessGDC.find_clinical(file_list, clinical_option) == True, "ERROR"


    def test_find_clinical(self):
        file_list = ["nationwidechildrens.org_clinical_patient_v1.2_skcm.txt", 
                     "nationwidechildrens.org_clinical_drug_skcm.txt",
                     "nationwidechildrens.org_clinical_treatment_skcm.txt"]
        
        clinical_option = "nationwidechildrens.org_clinical_patient_skcm.txt"
    
        assert ProcessGDC.find_clinical(file_list, clinical_option) == False, "ERROR"


    def test_download_data(self):
         gdc_query = GDCQuery(project = "TCGA-SKCM", legacy=True, data_category = "Gene expression", data_type = "Gene Expression Quantification",
                        experimental_strategy = "RNA-Seq", normalized = True)
         
         with self.assertRaises(SystemExit) as cm:
             ProcessGDC.download_data(gdc_query, os_path = "D:\TFG_archivos\info_y_copias\O\TCGA-SKCM")
    
         self.assertEqual(cm.exception.code, 0)


    def test_download_data1(self):
        gdc_query = GDCQuery(project = "TCGA-SKCM", legacy=True, data_category = "Gene expression", data_type = "Gene Expression Quantification",
                        experimental_strategy = "RNA-Seq", normalized = True)
        
        assert ProcessGDC.download_data(gdc_query) == -1, "ERROR"


    def test_download_data2(self):

        gdc_query = GDCQuery(project = "TCG", legacy=True, data_category = "Gene expression", data_type = "Gene Expression Quantification",
                        experimental_strategy = "RNA-Seq", normalized = True)
         
        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.download_data(gdc_query)
        
        self.assertEqual(cm.exception.code, 0)


    def test_download_data3(self):

        gdc_query = GDCQuery(project = "TCGA-UCS", legacy=True, data_category = "Gene", data_type = "Gene Expression Quantification",
                        experimental_strategy = "RNA-Seq", normalized = True)
         
        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.download_data(gdc_query)
        
        self.assertEqual(cm.exception.code, 0)

    
    def test_download_data4(self):

        gdc_query = GDCQuery(project = "TCGA-LAML", data_category = "Transcriptome Profiling", data_type = "Gene Expression Quantification",
                        experimental_strategy = "RNA-Seq", normalized = True)
         
        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.download_data(gdc_query)
        
        self.assertEqual(cm.exception.code, 0)


    def test_download_clinical(self):

        with self.assertRaises(SystemExit) as cm:
            ProcessGDC.download_clinical_data("TCGA-UCS", "patient_v12.0")
        
        self.assertEqual(cm.exception.code, 0)
    

# run the test
unittest.main()