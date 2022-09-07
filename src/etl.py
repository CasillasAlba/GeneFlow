# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)
"""


import csv
import math
import shutil
import gzip
from src import utils as ut
from src import processing as pro
from src.objects import FileProject
from src.objects import DataProject



# ETL == Extract, transform and load 
class ETL():
    
    endpt_base = ""
    endpt_status = ""
    endpt_projects = ""
    endpt_files = ""
    endpt_data = ""
    clinical_filename = ""
    clinical_options = []
    
    def __init__(self):
      """Constructor method
      """ 

    @staticmethod 
    def get_endpt_base():
        return ETL.endpt_base
    @staticmethod
    def get_endpt_files():
        return ETL.endpt_files   
    @staticmethod
    def get_endpt_status():
        return ETL.endpt_status
    @staticmethod
    def get_endpt_data():
        return ETL.endpt_data  
    @staticmethod
    def get_endpt_projects():
        return ETL.endpt_projects       
    @staticmethod
    def get_clinical_filename():
        return ETL.clinical_filename  
    @staticmethod
    def get_clinical_options():
        return ETL.clinical_options  
        
    @staticmethod
    def download_file(respFile, path_file):
        
        with open(path_file, "wb") as output_file:
            
            for block in respFile.iter_content(1024):
                
                output_file.write(block)
    
    # Creates a thread that executes the download_file function with the given arguments
    @staticmethod
    def thread_download_file(respFile, path_file):
        ut.Thread(target=ETL.download_file, args=(respFile, path_file)).start()


    # These methods won't be implemented because ETL will be the parent class
    # So the children class can implement their own methods.
    @staticmethod
    def get_projects(self):
        raise NotImplementedError("Must override get_projects")
        
    def search_project(self, ident):
        raise NotImplementedError("Must override search_project")
        
    def check_parameter(self, endpt, param, value, list_filter):
        raise NotImplementedError("Must override check_parameter")
        
    def check_data_type(self, endpt, filters, data_type):
        raise NotImplementedError("Must override check_data_type")
        
    def form_filter(self, fils, nameFilter, valueFilter, fullNameFilter):
        raise NotImplementedError("Must override form_filter")
    
    @staticmethod
    def get_query(self, query_object):
        raise NotImplementedError("Must override get_query")
        
    @staticmethod   
    def get_clinical_query(proj, legacy = False):
        raise NotImplementedError("Must override get_clinical_query")
        
    @staticmethod    
    def download_data(self, query_object, os_path = None):
        raise NotImplementedError("Must override download_data")
        
    @staticmethod
    def download_clinical_data(project_name, clinic_info, legacy = False, os_path = None):
        raise NotImplementedError("Must override download_clinic_data")
      
    @staticmethod
    def read_file_rna(name_tsv, col):
        raise NotImplementedError("Must override read_file_rna")
        
    @staticmethod
    def read_rna(dir_path, save = True):
        raise NotImplementedError("Must override read_rna")
        
    @staticmethod  
    def read_clinical(name_tsv, sep):
        raise NotImplementedError("Must override read_clinical")


class ProcessGDC(ETL):

    ETL.endpt_base = "https://api.gdc.cancer.gov/"
    ETL.endpt_status = ETL.endpt_base + "status"
    ETL.endpt_projects = ETL.endpt_base + "projects"
    ETL.endpt_files = ETL.endpt_base + "files"
    ETL.endpt_data = ETL.endpt_base + "data"
    ETL.clinical_filename = "nationwidechildrens.org_clinical_"
    ETL.clinical_options = ["drug", "patient", "nte", "radiation", "follow_up", "omf"]

    # Unique parameters of the child
    # Not all data sources will have a legacy parameter.
    endpt_legacy = ETL.endpt_base + "legacy/files"
    data_legacy = False

    def __init__(self):
        """Constructor method
        """ 
    
    @staticmethod
    def get_endpt_base():
        """Returns a string with the URL to search in the TCGA base
        
        :return: The URL to search in the GDC base.
        :rtype: str
        """
        return ProcessGDC.endpt_base

    @staticmethod
    def get_endpt_legacy():
        """Returns a string with the URL to search in the TCGA Legacy Archive
        
        :return: The URL to search in the GDC Legacy Archive.
        :rtype: str
        """
        return ProcessGDC.endpt_legacy
    
    @staticmethod
    def get_data_legacy():
        """Returns a boolean that indicates if the data is from the GDC
            Legacy Archive or from the GDC Data Portal.
        
        :return: 'True' to search for data in the GDC Legacy Archive,
            and 'False' otherwise.
        :rtype: bool
        """
        return ProcessGDC.data_legacy


    @staticmethod
    def set_data_legacy(legacy):
        """Set if data comes from the GDC Legacy Archive or not.
        
        :param legacy: 'True' to search for data in the GDC Legacy Archive,
            and 'False' otherwise.
        :type legacy: bool 
        """
        ProcessGDC.data_legacy = legacy
    
    
    @staticmethod
    def get_projects():
        """Gets all available GDC projects
            
            :return: List of available projects in the GDC Data Portal.
            :rtype: list
        """
        
        params = {
            "size" : 1000,
            "format" : "json"
        }
          
        data = ut.get_json_query(ETL.get_endpt_projects(), params)
        
        projects = data["data"]["hits"]
          
        project_identifiers = {}
          
        for p in projects:
            project_identifiers[p["project_id"]] = p["name"]
      
        return project_identifiers
    
    
    @staticmethod
    def search_project(ident):
        """Search the project by the input "ident" variable. It could be the identifier
            or the complete name of the cancer. If exists, the ID will be returned.
            
            :param ident: Identificator (ID/Name) to search the project.
            :type ident: str
            :return: The identifier of the project. E.g: TCGA-BRCA.
            :rtype: str
        """
        
        projects = ProcessGDC.get_projects() # Return complete name of the cancer        

        # None if doesn't exists 
        # search for key and return value
        id_ = projects.get(ident) # Adrenocortical Carcinoma , Osteosarcoma, Skin Cutaneous Melanoma...
    
       
        if (id_ == None):
                
            val_list = list(map(lambda x: x.lower(), projects.values()))
                       
            try:
                pos = val_list.index(ident)                
                    
                id_ = list(projects.keys())[pos] # Convert complete name to acronym     
    
            except:
                
                ut.sys.exit(0)
        else:
            
            id_ = ident
            
    
        return id_
    
    
    @staticmethod
    def check_parameter(endpt, param, value, filters):
        """Check if the parameter selected is available with the actual query.
            
            :param endpt: URL where the request is going to be done.
            :type endpt: str
            :param param: Name of the parameter to check.
            :type param: str
            :param value: Value of the parameter to check.
            :type value: str
            :param filters: current list of filters.
            :type filters: list

            :return: The value of parameter with the correct format to add 
                it to the query.
            :rtype: str
        """
        
        # Search parameters
        params = {
            'facets': param,
            "filters": ut.json.dumps(filters),
            'from':0, 'size':0,
            "pretty" : "true"
        }
             
        response = ut.requests.get(endpt, params = params)     
                
        mapa = response.content
        
        # Transform to json format (dict)
        data = ut.json.loads(mapa)
                
        datos = data["data"]["aggregations"][param]["buckets"]
        
        param_original = []
        
        param_results = []
        
        # get all available parameters with the actual filters
        for d in datos:
            param_original.append(d["key"])
            param_results.append(d["key"].lower())
      
            
        # If we only have a single value, we search it in the list
        # If we have several values, we have to check if each element is in param_results list
        if type(value) is not list:
            pos = ut.find_parameter(value, param_results, param)
    
        else:
            for va in value:
                pos = ut.find_parameter(va, param_results, param)

        return param_original[pos]
    

    @staticmethod
    def check_data_type(endpt, filters, data_type):
        """Check the data type selected. We need to have a single data type to 
            proceed with the download. So, data type cannot be a list.
            If None, we have to check if all the filters selected return 
            only one data type to download.
            
            :param endpt: URL where the request is going to be done.
            :type endpt: str
            :param filters: current list of filters.
            :type filters: list
            :param data_type: Data Type to check.
            :type data_type: str
        """
        
        if data_type != None:
            
            if type(data_type) is list:
                
                print("We can only download one data type. Please use 'Data Type' argument to filter results.")
                
                ut.sys.exit(0)
        else:
            
            params = {
                'facets': "data_type",
                "filters": ut.json.dumps(filters),
                'from':0, 'size':0,
                "pretty" : "true"
            }
            
            response = ut.requests.get(endpt, params = params)
            
            mapa = response.content
                
            # Transform data to JSON format
            data = ut.json.loads(mapa)
            
            datos = data["data"]["aggregations"]["data_type"]["buckets"]
            
            param_results = []
            
            for d in datos:
                param_results.append(d["key"].lower())
            
            # len(param_results) > 1 means there are more than one data type with the filters selected, so the program
            # finishes cause we only download data with a single data type.
            if len(param_results) > 1:
                
                print("We can only download one data type. Please use 'Data Type' argument to filter results.")
                
                ut.sys.exit(0)


    @staticmethod
    def form_filter(filters, nameFilter, valueFilter, fullNameFilter, endpt_):
        """Creates the filter to do the query that gets the data.
            If the new value for the query is available, it will be added to
            the filter's list.
            
            :param filters: Current list with the filters.
            :type filters: list
            :param nameFilter: Name of the filter to be checked and added.
            :type nameFilter: str
            :param valueFilter: Value of the filter to be checked and added.
            :type valueFilter: str
            :param fullNameFilter: Name of the filter to be added plus the path 
                where it is searched in the API.
            :type fullNameFilter: str 
        """
        
        # When an object of type list is passed to a numpy array,
        # it is passed by reference automatically, so its value is pounded.
        
        # Check if the parameter exist. If not, the program will stop the execution
        param_original = ProcessGDC.check_parameter(endpt_, nameFilter, valueFilter, filters)
            
        opin = ut.create_opin((fullNameFilter, param_original))
        
        filters["content"].append(opin)
        
        
        
    @staticmethod
    def regex_file_name(proj_id, clinical_option, file_name):
        """Finds the clinical option even if it is a new version of the file.
            The version of the file is controlled by a regex expression. 
            E.g:  _omf_v1.0 or _omf_v12.9 can be part of the file name
            
            :param proj_id: ID of the project. E.g: if the project is "TCGA-SKCM", the id
                will be "skcm".
            :type proj_id: str
            :param clinical_option: Clinical option to search in the regex expression.
            :type clinical_option: str
            :param file_name: File to check if it has the clinical option in its name, even
                if it has a version pattern.
            :type file_name: str
            :return: 'True' if the clinical option has been found and 'False' otherwise.
            :rtype: bool
        """
        
        middle_nm = ETL.get_clinical_filename() + clinical_option
        
        pattern = middle_nm + "(_v[0-9]{1,3}.[0-9]{1})?" + "_" + proj_id + ".(txt|csv|tsv)$"
        
        result = ut.re.match(pattern, file_name.lower())
        
        if result:
            return True
        else:
            return False
    

    @staticmethod
    def find_clinical(file_list, clinical_option):
        """ Finds if the selected clinical option is one of the available list 
            of files.
            First, we only kept with those files from the list of files that match
            a pattern where it is searched tyhe word "clinical" in the name of the file.
            Then, in a new list with only the clinical files, search for a file that have
            the specified clinical option.
         
            :param file_list: List of files to search the clinical option's file.
            :type file_list: list
            :return: 'True' if the clinical option has been found and 'False' otherwise.
            :rtype: bool
        """
        
        # Any ASCII Character (0 to 127)
        pattern = "^[\x00-\x7F]*" + "_clinical_[\x00-\x7F]*(.txt|.csv|.tsv)$"
        
        list_clinical = []
        
        for a in file_list:
            
            if ut.re.match(pattern, a.lower()):
                list_clinical.append(a)    
        
        # If between all list_clinical elements, one of them have the clinical_option
        # in the name, the file will be downloaded
        for cln in list_clinical:
            # search substring on list
            if ut.re.search(clinical_option, cln):
                return True
        
        return False
    
    
    @staticmethod
    def write_to_dataframe(list_, file, col, entity_id, legacy):
        """Creates a list of the DataFrame's lines, so it allows multithreading
            while reading the file.
            
            :param list_: List to append DataFrame's content. It is modified by reference.
            :type list_: list
            :param file: file to be read.
            :type file: str
            :param col: Name of the column to be read. If legacy is 'True', 
                it is set to the second column of the DataFrame; if legacy is
                'False', column is set to 'unstranded'.
            :type col: str
            :param legacy:'True' to search for data in the GDC Legacy Archive,
                and 'False' otherwise; defaults to 'False'.
            :type legacy: bool 
        """
        
        if(legacy):
            file_content = ProcessGDC.read_file_rna_legacy(file, col)
            file_content = file_content.rename(columns = {file_content.columns[0] : entity_id})
            
        else:
            file_content = ProcessGDC.read_file_rna(file, col)
            file_content = file_content.rename(columns = {col : entity_id})
    
        list_.append(file_content)



###########################################################################
#       
#                          GET QUERY FUNCTIONS
# 
###########################################################################


    @staticmethod
    def get_query(query_object):
        """ It creates the correct query using the data specified in the input DataObject
            and gets the data that it is going to be downloaded.
            To do it, it is checked each parameter availability taking into account the filter
            selected.
            If the parameter is available, it is added to the filter's list before checking
            the next parameter. 
            Once all filters are set, we get the request.
         
            :param query_object: GDCQuery that specified all the fields to get the data desired.
            :type query_object: GDCQuery
            :return: The response of the request, in case all parameters are available.
            :rtype: requests.models.Response
        """
    
        # list of the fileds that will be returned by the query. We avoid the rest of the information
        lfields = ["file_id", "data_format", "file_name", "submitter_id", "data_category", "type", "access",
                   "file_size", "created_datetime", "md5sum", "updated_datetime", "data_type", "state",
                   "experimental_strategy", "version", "data_release", "associated_entities.entity_submitter_id"]
        
        lfields = ",".join(lfields)
      
        filters = {"op" : "and", "content" : []}
    
    
        if(query_object.get_legacy() == True):
            
            endpt = ProcessGDC.get_endpt_legacy()
            ProcessGDC.set_data_legacy(True)
            
        else:
            
            endpt = ProcessGDC.get_endpt_files()
            ProcessGDC.set_data_legacy(False)


        # Add project to filter. 
        # It is not needed to check this param because at this point it is sure that the project id is correct
        print("Project " + str(query_object.get_project()))
        
        opin = ut.create_opin(("cases.project.project_id", query_object.get_project()))
        
        filters["content"].append(opin)
        
        print("Checking if the parameters are correct...\n")


        # We check each of the params introduced by the user and add them into the filter if they are correct
        # Taking into account we will analyse RNA-Seq data,this parameter will be mandatory for the user
        #  and it will be the first one to be checked
        if(query_object.get_experimental_strategy() != None):
            
            ProcessGDC.form_filter(filters, "experimental_strategy", query_object.get_experimental_strategy(), "files.experimental_strategy", endpt)
            
        if(query_object.get_data_category() != None):
            
            ProcessGDC.form_filter(filters, "data_category", query_object.get_data_category(), "files.data_category", endpt)
                  
        if(query_object.get_data_type() != None):
            
            ProcessGDC.form_filter(filters, "data_type", query_object.get_data_type(), "files.data_type", endpt)
                            
        if(query_object.get_workflow_type() != None and query_object.get_legacy() != True):
            
            ProcessGDC.form_filter(filters, "analysis.workflow_type", query_object.get_workflow_type(), "files.analysis.workflow_type", endpt)                    
            
        if(query_object.get_access() != None):
            
            ProcessGDC.form_filter(filters, "access", query_object.get_access(), "files.access", endpt)
            
        if(query_object.get_platform() != None):
            
            ProcessGDC.form_filter(filters, "platform", query_object.get_platform(), "files.platform", endpt)
            
        if(query_object.get_data_format() != None):
            
            ProcessGDC.form_filter(filters, "data_format", query_object.get_data_format(), "files.data_format", endpt)
                                          
        if(query_object.get_sample_type() != None):
            
            ProcessGDC.form_filter(filters, "cases.samples.sample_type", query_object.get_sample_type(), "files.cases.samples.sample_type", endpt)

        # Finally, we check if our filters only return a sigle type of data. 
        ProcessGDC.check_data_type(endpt, filters, query_object.get_data_type()) 


        params = {
            "filters": ut.json.dumps(filters), 
            "facets": "file_size", # to get file size of the query
            "fields": lfields,
            "size": "900000", # Used 900000 to download all the data available for the query
            "pretty": "true"
        }


        # We try twice to do the query just to controll a posible timeout problem.
        for i in range(2):  
            
            try:
                response = ut.requests.get(endpt, params=params,  timeout = 600)
                            
                return(response)
            
            except:
                
                if (i == 0):
                    print("We will retry to access GDC!")
                    
                else:
                    print("Timeout error trying the query.")


    # Get clinical info for "proj" project and "clinic" clinical value               
    @staticmethod
    def get_clinical_query(project_name, legacy):
        """Query in the TCGA API to get the clinical info linked to a 
            specific project.
            
            :param project_name: Name of the project.
            :type project_name: str
            :param legacy: 'True' to search for data in the GDC Legacy Archive,
                and 'False' otherwise; defaults to 'False'.
            :type legacy: bool 
        """
    
        filters = {"op" : "and", "content" : []}
    
        if(legacy):
            endpt = ProcessGDC.get_endpt_legacy()
            
        else:
            endpt = ProcessGDC.get_endpt_files()
            
                
        # Add project to filter 
        
        print("Project " + str(project_name))
        
        opin = ut.create_opin(("cases.project.project_id", project_name))
        
        filters["content"].append(opin)
        
        print("Checking clinical information...\n")
        
        ProcessGDC.form_filter(filters, "data_category", "Clinical", "files.data_category", endpt)
                                          
        params = {
            "filters": ut.json.dumps(filters),
            "fields" : "file_name",
            "size": "90000",
            "pretty": "true"
        }
        
        # We try twice to do the query just to controll a posible timeout problem.
        for i in range(2):  
            
            try:
                response = ut.requests.get(endpt, params=params,  timeout = 600)
                
                return(response)
            
            except:
                
                if (i == 0):
                    print("We will retry to access GDC!")
                    
                else:
                    print("Timeout error trying the query.")



###############################################################################
        
#                           DOWNLOAD DATA FUNCTIONS

###############################################################################


    @staticmethod
    def download_data(query_object, os_path = None):
        """ Download the data with the characteristics specidied in the DataObject object.
            Before downloading the data, it is checked if all fields have a correct value and
            existing information.
            A folder with the ID of the cancer will be created to save the data. It will also be
            saved a manifest.txt file with the list of the ID of the files and their Entity ID.
            The Entity ID is useful to match samples with clinical information downloaded from the 
            portal.
            All the data is downloaded using the TCGA API via requests.
         
            :param query_object: GDCQuery object with the features of the data that will be downloaded.
            :type query_object: GDCQuery
            :param os_path: Directory path were the file will be saved.Defaults to None, so
                it will be saved in the current directory.
            :type os_path: str, optional
        """
                
        # Check if the path exists
        if os_path != None and not (ut.os.path.isdir(os_path)):
            
            print("The destination path does not exist") 
            
            ut.sys.exit(0)
        
        # Check the status of the server
        ut.check_server_status(ETL.get_endpt_status())
        
        # Search the project in GDC database.
        # If -1 is returned, the project is not found --- show list of available projects
        print("Searching in GDC database")
        
        # Search the project in the db and update the attribute
        query_object.set_project(ProcessGDC.search_project(query_object.get_project()))
        
        if(query_object.get_project() != -1):
            
            # Create the folder to save the data if it doesn't exist.
            # If the folder exists, it might be because the data is already downloaded.
            try:
                if os_path == None:
                    
                    destino = query_object.get_project()
                    
                    ut.os.mkdir(destino) 
                    
                else:
                    
                    destino = os_path + ("/") + query_object.get_project()
                    
                    ut.os.mkdir(destino)
                
            except:
                                
                content = next(ut.os.walk(destino))

                # If the condition is True, files of the project have been downloaded
                # and it might also be downloaded clinical files
                if len(content[1]) > 0: 
                    
                    print("Project: " + str(query_object.get_project()) + " already exists in the current directory.")    
                    return -1  # The function finishes but not the whole program            
                
            # Get the query with filters elected
            print("Accessing GDC. This might take a while...\n")
            
            response = ProcessGDC.get_query(query_object)
            
            content = response.content
    
            data = ut.json.loads(content) # Transform to json format (dict)
    
            # We get the data and the total size of the query, which is the sum of the size of each result in bytes
            datos = data["data"]["hits"] # Access to third level
            
            # We add the information of each file to the FilesTCGA class.
            # If the try fails, in the case that one of the parameters doesn't ecists, we add only the essential fields of the files
            datas_downloaded = DataProject()
            
            datas_downloaded.set_project_name(query_object.get_project())
            
            # Create a manifest file
            # The manifest file will save the file name and its corresponding Entity ID
            # This will be useful to read the downloaded data correctly
            manifest = open(destino + "/manifest.txt", "w")
            manifest.write("File_Name\tEntity_ID\n")
                  
            for d in datos: 
                
                try:
                    if query_object.get_normalized():
                        
                        if d["file_name"].find(query_object.get_file_type()) != -1:

                            fproj = FileProject(d["id"], d["data_format"], d["access"], d["file_name"], d["submitter_id"],
                                                d["data_category"], d["type"], d["file_size"], d["created_datetime"], d["md5sum"],
                                                d["updated_datetime"], d["data_type"], d["state"], d["experimental_strategy"], 
                                                d["version"], d["data_release"], d["associated_entities"][0]["entity_submitter_id"])
                            
                            datas_downloaded.add_file(fproj)
                            
                            datas_downloaded.plus_size_download(d["file_size"] / math.pow(10,6)) #  # Get the file size
                            
                            manifest.write(fproj.get_file_name() + "\t" + fproj.get_entity_id() + "\n")
                    
                    else:

                        fproj = FileProject(d["id"], d["data_format"], d["access"], d["file_name"], d["submitter_id"],
                                            d["data_category"], d["type"], d["file_size"], d["created_datetime"], d["md5sum"],
                                            d["updated_datetime"], d["data_type"], d["state"], d["experimental_strategy"], 
                                            d["version"], d["data_release"], d["associated_entities"][0]["entity_submitter_id"])

                        datas_downloaded.add_file(fproj)

                        datas_downloaded.plus_size_download(d["file_size"] / math.pow(10,6))  # Get the file size
                        
                        manifest.write(fproj.get_file_name() + "\t" + fproj.get_entity_id() + "\n")

                except:
                    
                    if query_object.get_normalized():

                        if d["file_name"].find(query_object.get_file_type()) != -1:

                            fproj = FileProject(d["id"], data_format = d["data_format"], file_name = d["file_name"], 
                                                submitter_id = d["submitter_id"], data_category = d["data_category"], 
                                                file_size = d["file_size"], data_type = d["data_type"],
                                                entity_id = d["associated_entities"][0]["entity_submitter_id"])
                                             
                            
                            manifest.write(fproj.get_file_name() + "\t" + fproj.get_entity_id() + "\n")

                            datas_downloaded.add_file(fproj)

                            datas_downloaded.plus_size_download(d["file_size"] / math.pow(10,6))  # Get the file size


                    else:
                        fproj = FileProject(d["id"], data_format = d["data_format"], file_name = d["file_name"], 
                                            submitter_id = d["submitter_id"], data_category = d["data_category"], 
                                            file_size = d["file_size"], data_type = d["data_type"],
                                            entity_id = d["associated_entities"][0]["entity_submitter_id"])
                                         
                        manifest.write(fproj.get_file_name() + "\t" + fproj.get_entity_id() + "\n")

                        datas_downloaded.add_file(fproj)

                        datas_downloaded.plus_size_download(d["file_size"] / math.pow(10,6))  # Get the file size
                        
            manifest.close()
                
            # Create a post reuest to donwload multiple files
            # https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/#post-request-to-download-multiple-files
            # With the id of the file, we do a post request to download its information.
            # For each file, a folder with its file name is created in order to save the related info.
            
            datas_downloaded.set_dir_path(destino)
            destino = destino + ("/")           
            i = 0
            size_mb = 0
            total_files = datas_downloaded.get_total_files()
            total_size = round(datas_downloaded.get_size_download(), 2)

            print("Downloading data for project " + str(query_object.get_project()) + ".")
            
            ut.print_progress_bar(i, total_files, size_mb , total_size)
            
            for f in datas_downloaded.get_files():
                
                path = destino + f.get_id()

                ut.os.mkdir(path)

                path_fl = datas_downloaded.formatted_path(f)
                
                size_mb = size_mb + (f.get_file_size() / math.pow(10,6)) # to show the size of the data donwloaded.

                size_mb_rd = round(size_mb, 2)

                i = i + 1 # Update progress bar
    
                resp = ut.requests.post(ETL.get_endpt_data(), data = ut.json.dumps({"ids" : f.get_id()}), headers= {"Content-Type" : "application/json"}, stream=True)
                
                ETL.thread_download_file(resp, path_fl) # Download the data with multithreadin to increase the speed

                ut.print_progress_bar(i, total_files, size_mb_rd , total_size)
                
        else:
            
            project_identifiers = ProcessGDC.get_projects()
            
            print("Project: " + str(query_object.get_project()) + " was not found. Availables projects are:\n ")    
    
            for k, v in project_identifiers.items():
    
                label, value = k, v
                
                print("{:<15} {:<10}".format(label, value))
                
            print("Please set a valid project argument.")
            #print(ut.fg('light_red') + "Please set a valid project argument.")
                
            ut.sys.exit(0)



    @staticmethod
    def download_clinical_data(project_name, clinic_info, legacy = False, os_path = None):
        """ Download clinical data from a specified project. A folder with the name of the
            project will be created if it does not already exist. If it exists, it will be
            checked if the clinical information has been previously downloaded.
         
            :param project_name: Name of the project.
            :type project_name: str
            :param clinic_info: Type of clinical information that will be downloaded. 
                It will be downloaded the latest version found. E.g: "patient" or "drug".
            :type clinic_info: str
            :param legacy: True to search for data in the GDC Legacy Archive,
                and 'False' otherwise; defaults to 'False'.
            :type legacy: bool
            :param os_path: Directory path were the file will be saved. Defaults to None, so
                it will be saved in the current directory.
            :type os_path: str, optional
        """
        
        # Check if the path exists
        if os_path != None and not (ut.os.path.isdir(os_path)):
            
            print("The destination path does not exist") 
            
            ut.sys.exit(0)

        # Check the status of the server
        ut.check_server_status(ETL.get_endpt_status())   
        
        # Search the project in GDC database.
        # If .1 is returned, the project is not found --- show list of available projects
        print("Searching in GDC database")
        
        # Search the project in the db and update the attribute
        project_id = ProcessGDC.search_project(project_name)
        
        if(project_id != -1):
            # dpr.check_element_on_list(clinic_info, ETL.get_clinical_options())
            pro.CheckElement(clinic_info).apply(ETL.get_clinical_options())
            
            # Create the folder to save the data if it doesn't exist.
            # If the folder exists, it might be because the data is already downloaded.

            try:
                if os_path == None:

                    destino = project_id

                    ut.os.mkdir(destino)
                    
                else:
                    
                    destino = os_path + ("/") + project_id
                    
                    ut.os.mkdir(destino)
                    
            except:

                content = next(ut.os.walk(destino))
                
                # If files are found, it is needed to check if the clinic option has been downloaded
                # e.g: it may be downloaded clinical_drug but not clinical_patient  
                found = ProcessGDC.find_clinical(content[2], clinic_info)

                if found:
                
                    print("Clinical Information: " + clinic_info + " already exists in the current directory.")
    
                    return -1 # The function finishes but not the whole program


            # Get the query with filters elected
            print("Accessing GDC. This might take a while...\n")
                                        
            response = ProcessGDC.get_clinical_query(project_id, legacy)
                            
            content = response.content
              
            data = ut.json.loads(content) # Transform to json format (dict)                      
                            
            # We get the data and the total size of the query, which is the sum of the size of each result in bytes
            datos = data["data"]["hits"] # Access to third level
             
            # We need the second part of the name, e.g: TCGA-SKCM => skcm
            low_proj = project_id.split("-")[1].lower()
                 
            # If 0 is returned, a file for the indicated clinical data was not found
            file_search = list(filter(lambda fl: ProcessGDC.regex_file_name(low_proj, clinic_info, fl['file_name']) == True, datos))
                             
            if len(file_search) == 0:
                                
                print("Clinical File: " + clinic_info + " was not posible to find")
                  
                ut.sys.exit(0)
                         
                
            else:

                contenido = next(ut.os.walk(destino))
                 
                if file_search[0]["file_name"] in contenido[2]:
                 
                    print("Clinical File: " + str(file_search[0]["file_name"]) + " already exists in the current directory.")
                 
                    ut.sys.exit(0)
                 
                
                print("Downloading clinical information: " + str(clinic_info) + ".")
                 
                path = destino + str("/") + file_search[0]["file_name"]
                 
                resp = ut.requests.post(ETL.get_endpt_data(), data = ut.json.dumps({"ids" : file_search[0]["id"]}), headers= {"Content-Type" : "application/json"}, stream=True)
                
                ETL.thread_download_file(resp, path)
            
        else:
            
            project_identifiers = ProcessGDC.get_projects()
            
            print("Project: " + str(project_id) + " was not found. Availables projects are:\n ")
    
    
            for k, v in project_identifiers.items():
    
                label, value = k, v
                
                print("{:<15} {:<10}".format(label, value))
                
            print("Please set a valid project argument.")
            #print(ut.fg('light_red') + "Please set a valid project argument.")
                
            ut.sys.exit(0)
    

    @staticmethod
    def download_genecode(url_gtf = "https://api.gdc.cancer.gov/data/be002a2c-3b27-43f3-9e0f-fd47db92a6b5", name_gtf = "genecode_v36"):
        """Download genecode to create a data structure with information about
            specific genes. It will allow building normalization functions.

            :param url_gtf: URL where the genecode is hosted. By default, the program download from GDC and its 36th version.
            :type url_gtf: str
            :param name_gft: Name of the file (WITHOUT extension). Its extension must be .gtf
            :param name_gft: str
        """

        r = ut.requests.get(url_gtf, stream=True)
        full_name = name_gtf + ".gtf"

        if r.status_code == 200:
        
            print("Downloading gene model...")
        
            with open(full_name, 'wb') as f:
                r.raw.decode_content = True  # just in case transport encoding was applied
                gzip_file = gzip.GzipFile(fileobj=r.raw)
                shutil.copyfileobj(gzip_file, f)
            
            print("\nGene Model Downloaded\n")

        else:
            print("Error trying to download gene model")
            ut.sys.exit(0)


    @staticmethod
    def create_gene_model(path_gtf = "genecode_v36.gtf", path_save = "genecode_v36.csv"):
        """Create a DataFrame that allow an easy treatment with information about Genes.
            The function will create a TSV readable with features of Genes. If CSV exist, there is
            not neccesary to call this function, only read it.

            :param path_gtf: Path where genecode with GTF format is located
            :type path_gtf: str
            :param path_save: Path where the CSV result will be save
            :param path_save: str
        """

        # Only necessary to create the genecode model
        from rpy2.robjects.packages import importr
        import rpy2.robjects as ro

        utils = importr('utils')
        utils.install_packages("BiocManager")

        BiocManager = importr("BiocManager")
        BiocManager.install("GenomicFeatures")

        GenomicFeatures = importr("GenomicFeatures")

        # It only takes a minute to run (3M lines)
        gtf_txdb = GenomicFeatures.makeTxDbFromGFF(path_gtf)
        gene_list = GenomicFeatures.genes(gtf_txdb)

        dataf = ro.r['data.frame'](gene_list)

        # R's Function to write on a new File
        write_csv = ro.r('write.csv') # Generate de R's Function to use then

        write_csv(dataf, path_save)



    @staticmethod
    def read_genecode_csv(file_csv="../genecode_v36.csv", sep=","):
        """Read a CSV file that contains information about genes
            and return a DataFrame to work with it

            :param file_csv: Path where the genecode on CSV format is saved
            :type file_csv: str
            :param sep: Character that delimit its structure
            :type sep: str

            :return: A DataFrame with Gene Information
            :rtype: DataFrame
        """

        genemodel = ut.read_file(file_csv, sep=sep)

        columns_nm = pro.DataColumnames().apply(genemodel)
        
        if columns_nm[0] == "Unnamed: 0":
            
            genemodel = genemodel.iloc[:, 1:]
        
        return genemodel.set_index("gene_id")



###############################################################################
        
#                        READ DATA FUNCTIONS

###############################################################################


    @staticmethod      
    def read_file_rna_legacy(file_name, col, sep = "\t"):
        """ Read a single Rna-Seq file from The TCGA Legacy Archive.
            The first line is the column's names.
            An undefined number of rows are skipped while their first character
            is a '?'.
            The index is set to the first column, which is the gene_id.
            In this type of files the gene_id are made up of the ID of the gene and a number,
            separated by a "|". Only the ID of the gene is saved.
         
            :param file_name: File name to be read.
            :type dir_path: str
            :param col: Name of the column to be read. Set to 'unstranded'.
            :type col: str
            :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
            :type sep: str

            :return: DataFrame with file's data read.
            :rtype: DataFrame
        """
    
        header = ""
        data_file = []
        skip_line = True
        
        with open(file_name) as file:
            tsv_file = csv.reader(file, delimiter = sep)
        
            header = next(tsv_file)
          
            # ship the line while the first charcater is a "?"
            while(skip_line):
                
                line = next(tsv_file)
              
                if (line[0][0] != "?"):
                    
                    skip_line = False
                    
                    data_file.append(line)
              
                
            for line in tsv_file:
                
                data_file.append(line)        
        
        dFrame = ut.pd.DataFrame(data = data_file, columns=header)
       
        ids_ = list(dFrame.columns)[0]
        
        dFrame = dFrame.set_index(ids_) # Primary Key Columns
        
        # The gene_is strucutre is GeneID|Number, so only the ID of the gene is saved
        dFrame.rename(index=lambda rw : str(rw.split(sep="|")[0]), inplace=True)
          
        # It is col-1 because we set the index earlier, so if col = 1, now
        # it will be col = 0
        dFrame = dFrame.iloc[0:, col-1]

        return dFrame.to_frame()



    @staticmethod
    def read_file_rna(file_name, col, sep = "\t"):
        """ Read a single Rna-Seq file from The GDC portal. 
            The first line of this type of files is a comment that it is skipped.
            The second line will be the header of the data.
            Next lines are skipped / saved in a summary_file because they store 
            summary information of the file.
            The index is set to the first column, which is the gene_id.
         
            :param file_name: File name to be read.
            :type dir_path: str
            :param col: Name of the column to be read. Set to 'unstranded'.
            :type col: str
            :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
            :type sep: str

            :return: DataFrame with file's data read.
            :rtype: DataFrame
        """
        
        head_file = ""
        summary_file = {}
        data_file = []
        init_range = 6
        
        with open(file_name) as file:
            tsv_file = csv.reader(file, delimiter = sep)
        
            next(tsv_file) # Remove comment
        
            head_file = next(tsv_file)
        
            for i in range(2, init_range):
              
                summ_f = next(tsv_file)
                summary_file[summ_f[0]] = list(filter(None, summ_f[1:]))
              
            for line in tsv_file:
                data_file.append(line)
        
        
        dFrame = ut.pd.DataFrame(data = data_file, columns=head_file)
        
        ids_ = list(dFrame.columns)[0] # Gene_ID will be the column

        dFrame = dFrame.set_index(ids_) # Primary Key Columns
        
        # Check if the column name exists
        if col not in list(dFrame.columns):
            ut.sys.exit(0)
        
        dFrame = dFrame[col]
        
        return dFrame.to_frame()
                

    @staticmethod
    def read_rna(dir_path, sep = "\t", save = True):
        """ Read Rna-Seq files from The Cancer Genome Atlas (TCGA) portal.
            The file will be read in a different way depending on whether it is from 
            the Legacy Data portal or from the GDC Data portal.
            For the GDC portal, unstranded data is downloaded by default.
            For the Legacy Archive, it will be read the second column of the file
            that can be raw counts or normalized counts. While downloading the data, 
            the user can indicate the type of data to dowload by the parameter "normalized".
            
            :param dir_path: The path of the folder where the project has been downloaded. 
                By feault, the program search the folder in the current directory, so if 
                the project has been saved in a different folder, the user must indicate
                the full path.
            :type dir_path: str
            :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
            :type sep: str
            :param save: 'True' to keep the downloaded data in the folder and 'False' to remove
            the downloaded data after read it; defaults to 'True'.
            :type save: bool

            :return: DataFrame with the RNA-Seq information read.
            :rtype: DataFrame
        """
        
        # Check if the path exists
        if not (ut.os.path.isdir(dir_path)):
            
            print("The destination path of project does not exist.") 
            
            ut.sys.exit(0)
            
        print("Reading manifest...\n")
        
        # Read the associated manifest file to get Entity IDs
        try:
            manifest = ut.pd.read_csv(dir_path + "/manifest.txt", sep = sep)
            
            manifest = manifest.set_index("File_Name")
            
            print("Manifest read.\n")
            
        except:

            print("manifest.txt file does not exist.")
            
            ut.sys.exit(0)
            
      
        content = next(ut.os.walk(dir_path))
       
        subcontent = next(ut.os.walk(content[0] + "/" + content[1][0]))
        
        # We need to open one of the files in order to know if the files we are going to read are
        # from the legacy database or not. This is because the structure is different between both files.
        # If it is from the actual database, the first line will be a comment and it will start with a "#"
        # If legacy == False -> column unstranded will be downloaded by default.
        # If legacy == True -> there is only one column available. The name of the column will be different
        # depending on the nature of the data: raw_counts if it is raw data or normalized_results if it is normalized,
        # normalized or raw data is selected in the query with the "normalized" parameter.
        
        with open(subcontent[0] + "/" + subcontent[2][0]) as f:
            
            firstline = (f.readline().rstrip()).split(sep="\t")
        
        # If the first character of the line is an # => legacy == False
        if firstline[0][0] == "#":
            
            ProcessGDC.set_data_legacy(False)        
            # Unstranded data is read.
            col = "unstranded"
         
        else:
            
            ProcessGDC.set_data_legacy(True)            
            # The second (column = 1) is read.
            col = 1

                           
        entity_id = manifest.at[subcontent[2][0], "Entity_ID"]
        
        # DataFrame is formed
        
        if(ProcessGDC.get_data_legacy()):
            
            data_analysis = ProcessGDC.read_file_rna_legacy(subcontent[0] + "/" + subcontent[2][0], col, sep)
            data_analysis = data_analysis.rename(columns = {data_analysis.columns[0] : entity_id})
            
        else:
            
            data_analysis = ProcessGDC.read_file_rna(subcontent[0] + "/" + subcontent[2][0], col, sep) #subcontent[0] is the root
            data_analysis = data_analysis.rename(columns = {col : entity_id})
        
        
        content[1].pop(0) # delete first element 
                
        
        print("Reading the data...\n")
        
        thread_ = 0
        lista_df = [data_analysis]
        
        for subd in content[1]:
                
            subcontent = next(ut.os.walk(content[0] + "/" + subd))
            
            file = subcontent[0] + "/" + subcontent[2][0]
            
            entity_id = manifest.at[subcontent[2][0], "Entity_ID"]

            thread_ = ut.ThreadWithResult(target=ProcessGDC.write_to_dataframe, args=(lista_df, file, col, entity_id, ProcessGDC.get_data_legacy()))
                
            thread_.start()
            thread_.join()
            

        print("Joining the data... It might take a moment.\n")
        
        data_analysis = ut.pd.concat(lista_df, axis = 1)
        
        print("Data successfully read.\n")
        
        
        if save == False:
            
            print("Data will be removed after read the files")
            
            ut.ut.recursive_delete_data(dir_path)
            
        
        return data_analysis  


    @staticmethod
    def read_clinical(file_name, sep="\t"):
        """ Read a clinical file preivously downaloaded from The Cancer Genome Atlas (TCGA) portal.
        In TCGA clinical files, the first line is a comment that it is skipped. The two following lines
        have the header names; only the second header's line is kept, due to they have the same values
        but with different names.
        Returns a dataframe with the clinical information where the index is set to 'bcr_patient_barcode',
        that match with the Entity ID of the samples.
        
        :param file_name: The name of the clinical project to read. By default, the programn search the file
            in the actual path, so the location of the file is required to find the file. 
            E.g: "TCGA-BRCA/nationwidechildrens.org_clinical_patient_brca.txt".
        :type file_name: str
        :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
        :type sep: str

        :return: DataFrame with the clinical information read. The index has been set to
            'bcr_patient_barcode', matching with the Entity ID of the samples.
        :rtype: DataFrame
        """
    
        header_line = ""
        data_file = []
      
        try:
            # In TCGA, clinical files have two different headers, 
            # we will just save the second one
            with open(file_name) as file:
                tsv_file = csv.reader(file, delimiter=sep)
              
                next(tsv_file)  
        
                header_line = next(tsv_file)
        
                next(tsv_file) 
                  
                for line in tsv_file:
                    data_file.append(line)        
        except:
            print("The file: " + str(file_name) + " does not exist.")
            ut.sys.exit(0)
        
        
        dFrame = ut.pd.DataFrame(data = data_file, columns = header_line)
        
        dFrame = dFrame.set_index("bcr_patient_barcode") # Primary Key Columns
        
        return dFrame