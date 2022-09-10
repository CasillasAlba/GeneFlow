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

import pandas as pd
import src.dataobject as dobj
from src.dataobject import DataObject
import src.utils as ut

import unittest

# A unit test is a test that checks a single component of code, 
# usually modularized as a function, and ensures that it performs as expected.
# Unit tests in PyUnit are structured as subclasses of the unittest.TestCase class,
# and we can override the runTest() method to perform our own unit tests
# which check conditions using different assert functions in unittest.TestCase
# https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/


# The test based on unittest module
class TestDataObject(unittest.TestCase):
    
    def test_copy_object(self):
        
        data = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13"],
            "TCGA-WT-AB44": [716.0, 2077.0, 1790.0],
            "TCGA-XX-A899": [24.0, 20.0, 1021.0],
            "TCGA-XX-A89A": [617.0, 1588.0, 1216.0]
        }

        df = pd.DataFrame(data)

        df = df.set_index("gene_id")

        data_object = DataObject(df)

        df_copy = ut.copy_object(data_object)

        assert data_object, df_copy


    def test_dataobject(self):
        
        batch_counts = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 1021.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts = pd.DataFrame(batch_counts)

        batch_counts = batch_counts.set_index("gene_id")

        batch_clinical = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALJ", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "55262FCB-1B01-4480-B322-36570430C917", "427D0648-3F77-4FFC-B52C-89855426D647", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28", "2014-7-28", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE", "FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast", "Breast", "Breast"]
        }

        batch_clinical = pd.DataFrame(batch_clinical)

        batch_clinical = batch_clinical.set_index("bcr_patient_uuid")


        batch_data_object = DataObject(batch_counts, obs_ = batch_clinical)

        
        # EXPECTED RESULTS

        batch_counts_result = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 1021.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts_result = pd.DataFrame(batch_counts_result)

        batch_counts_result = batch_counts_result.set_index("gene_id")

        batch_clinical_result = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "55262FCB-1B01-4480-B322-36570430C917", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast", "Breast"]
        }

        batch_clinical_result = pd.DataFrame(batch_clinical_result)

        batch_clinical_result = batch_clinical_result.set_index("bcr_patient_uuid")


        assert batch_data_object.get_counts().equals(batch_counts_result) and batch_data_object.get_obs().equals(batch_clinical_result), "ERROR. Count Matrix or Obs are malformed"


    def test_dataobject2(self):
        
        batch_counts = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 1021.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts = pd.DataFrame(batch_counts)

        batch_counts = batch_counts.set_index("gene_id")

        batch_clinical = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALJ", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "55262FCB-1B01-4480-B322-36570430C917", "427D0648-3F77-4FFC-B52C-89855426D647", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28", "2014-7-28", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE", "FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast", "Breast", "Breast"]
        }

        batch_clinical = pd.DataFrame(batch_clinical)

        batch_clinical = batch_clinical.set_index("bcr_patient_uuid")
            
        batch_gene = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000288671.1", "ENSG00000000457.14", "ENSG00000000460.17"],
            "gene_name": ["TSPAN6", "TNMD", "AC006486.3", "SCYL3", "C1orf112"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding", "protein_coding", "protein_coding"]
        }

        batch_gene = pd.DataFrame(batch_gene)

        batch_gene = batch_gene.set_index("gene_id")


        batch_data_object2 = DataObject(counts_dframe = batch_counts, obs_ = batch_clinical, vars_ = batch_gene)


        # EXPECTED RESULTS

        batch_counts_result2 = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 150.0]
        }

        batch_counts_result2 = pd.DataFrame(batch_counts_result2)

        batch_counts_result2 = batch_counts_result2.set_index("gene_id")


        batch_clinical_result2 = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "55262FCB-1B01-4480-B322-36570430C917", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast", "Breast"]
        }

        batch_clinical_result2 = pd.DataFrame(batch_clinical_result2)

        batch_clinical_result2 = batch_clinical_result2.set_index("bcr_patient_uuid")


        batch_gene_result2 = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000460.17"],
            "gene_name": ["TSPAN6", "TNMD", "C1orf112"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding"]
        }

        batch_gene_result2 = pd.DataFrame(batch_gene_result2)

        batch_gene_result2 = batch_gene_result2.set_index("gene_id")


        assert batch_data_object2.get_counts().equals(batch_counts_result2) and batch_data_object2.get_obs().equals(batch_clinical_result2) and batch_data_object2.get_vars().equals(batch_gene_result2), "ERROR. Count Matrix or Obs or Vars are malformed"


    def test_counts_selection(self):

        batch_counts = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 1021.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts = pd.DataFrame(batch_counts)

        batch_counts = batch_counts.set_index("gene_id")

            
        batch_gene = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000288671.1", "ENSG00000000457.14", "ENSG00000000460.17"],
            "gene_name": ["TSPAN6", "TNMD", "AC006486.3", "SCYL3", "C1orf112"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding", "protein_coding", "protein_coding"]
        }

        batch_gene = pd.DataFrame(batch_gene)

        batch_gene = batch_gene.set_index("gene_id")


        batch_data_object3 = DataObject(batch_counts, vars_ = batch_gene)

        rows_list = ["ENSG00000000003.15", "ENSG00000000460.17"]

        batch_data_object3_selection = dobj.CountsSelection(rows_list).apply(batch_data_object3)


        # EXPECTED RESULTS

        batch_counts_result3 = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 205.0],
            "TCGA-3C-AALI": [24.0, 534.0],
            "TCGA-3C-AALK": [617.0, 150.0]
        }

        batch_counts_result3 = pd.DataFrame(batch_counts_result3)

        batch_counts_result3 = batch_counts_result3.set_index("gene_id")


        batch_gene_result3 = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000460.17"],
            "gene_name": ["TSPAN6", "C1orf112"],
            "gene_type": ["protein_coding", "protein_coding"]
        }

        batch_gene_result3 = pd.DataFrame(batch_gene_result3)

        batch_gene_result3 = batch_gene_result3.set_index("gene_id")


        assert batch_data_object3_selection.get_counts().equals(batch_counts_result3) and batch_data_object3_selection.get_vars().equals(batch_gene_result3), "ERROR. Count Matrix or Obs or Vars are malformed"


    def test_counts_projection(self):

        batch_counts = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALI": [24.0, 20.0, 1021.0, 534.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts = pd.DataFrame(batch_counts)

        batch_counts = batch_counts.set_index("gene_id")

        batch_clinical = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALJ", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "55262FCB-1B01-4480-B322-36570430C917", "427D0648-3F77-4FFC-B52C-89855426D647", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28", "2014-7-28", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE", "FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast", "Breast", "Breast"]
        }

        batch_clinical = pd.DataFrame(batch_clinical)

        batch_clinical = batch_clinical.set_index("bcr_patient_uuid")


        batch_data_object4 = DataObject(batch_counts, obs_ = batch_clinical)

        columns_list = ["TCGA-3C-AAAU", "TCGA-3C-AALK"]

        batch_data_object4_projection = dobj.CountsProjection(columns_list).apply(batch_data_object4)


        # EXPECTED RESULTS


        batch_counts_result4 = {
            "gene_id" : ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13", "ENSG00000000460.17"],
            "TCGA-3C-AAAU": [716.0, 2077.0, 1790.0, 205.0],
            "TCGA-3C-AALK": [617.0, 1588.0, 1216.0, 150.0]
        }

        batch_counts_result4 = pd.DataFrame(batch_counts_result4)

        batch_counts_result4 = batch_counts_result4.set_index("gene_id")


        batch_clinical_result4 = {
            "bcr_patient_uuid" : ["TCGA-3C-AAAU", "TCGA-3C-AALK"],
            "bcr_patient_barcode": ["6E7D5EC6-A469-467C-B748-237353C23416", "C31900A4-5DCD-4022-97AC-638E86E889E4"],
            "form_completion_date": ["2014-1-13", "2014-7-28"],
            "gender": ["FEMALE", "FEMALE"],
            "tumor_tissue_site": ["Breast", "Breast"]
        }

        batch_clinical_result4 = pd.DataFrame(batch_clinical_result4)

        batch_clinical_result4 = batch_clinical_result4.set_index("bcr_patient_uuid")


        assert batch_data_object4_projection.get_counts().equals(batch_counts_result4) and batch_data_object4_projection.get_obs().equals(batch_clinical_result4), "ERROR. Count Matrix or Obs are malformed"



# run the test
unittest.main()