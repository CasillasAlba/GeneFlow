# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

from genericpath import exists
import json
from src.processing import Task


class GDCQuery():
    """This is a conceptual class representation of a request of data
    from Genomic Data Commoms (GDC) portal (https://portal.gdc.cancer.gov/)
    by using the GDC API (https://gdc.cancer.gov/developers/gdc-application-programming-interface-api).
    
    :param project: Cancer name or ID to download. E.g: "TCGA-BRCA", 
        "Acute Myeloid Leukemia", "adrenocortical carcinoma" ...
    :type project: str 
    :param experimental_strategy: Experimental strategies used for molecular
        characterization of the cancer.
    :type experimental_strategy: str
    :param data_category: A high-level data file category. E.g "Raw Sequencing
        Data" or "Transcriptome Profiling", dafults to None
    :type data_category: str, optional
    :param data_type: Data file type. Data Type is more granular than Data Category.
        E.g: "Align Reads" or "Gene Expression Quantification", defaults to None.
    :type data_type: str, optional
    :param workflow_type: Bioinformatics workflow used to generate or harmonize
        the data file. E.g: "STAR - Counts", "BCGSC miRNA Profiling", defaults to None.
    :type workflow_type: str, optional
    :param legacy: The GCD Legacy Archive hosts unharmonized legacy dara from repositories
        that predate the GDC (e.g: CGHub). Legacy data is not actively maintained, processed, or
        harmonized by the GDC. Legacy is 'True' to search for data in the GDC Legacy Archive,
        and 'False' otherwise; defaults to 'False'.
    :type legacy: bool
    :param platform: Technolofical platform on which experimental data was produced.
        E.g: "Illumina Human Methylation 450", defaults to None. 
    :type platform: str, optional
    :param data_format: Format of the Data File, defaults to None.
    :type data_format: str, optional
    :param sample_type: Describes the source of a biospecimen used for a laboratory test. 
        E.g: "Primary Tumor", ["Primary Tumor", "Solid Tissue Normal"], defaults to None.
    :type sample_type: list, str, optional
    :param normalized: This parameter is only valid when legacy is set as 'True'.
        If 'True', normalized data from the GDC Legacy Archive will be downloaded, 
        and 'False' otherwise; defaults to 'False'.
    :type normalized: bool
    """
    
    def __init__(self, project ,experimental_strategy, data_category = None, data_type = None, 
                 workflow_type = None, legacy = False,  platform = None, 
                 data_format = None, sample_type = None, normalized = False):
        
        """Constructor method
        """   
        self.project = project
        self.data_category = data_category
        self.data_type = data_type
        self.workflow_type = workflow_type
        self.legacy = legacy
        # Direclty set as "open" because normal users cannot search in the "controlled" acces archives.
        self.access = "open" 
        self.platform = platform
        self.data_format = data_format
        self.experimental_strategy = experimental_strategy
        self.sample_type = sample_type
        self.normalized = normalized
        if self.normalized:
            self.file_type = "normalized_results"
        else:
            self.file_type = None


    """
    GET
    """

    def get_project(self):
        """Returns a string with the name of the project.
        
        :return: The name of the project.
        :rtype: str
        """
        return self.project
    
    def get_data_category(self):
        """Returns a string with the category of the data.
        
        :return: The Data Category.
        :rtype: str
        """
        return self.data_category
    
    def get_data_type(self):
        """Returns a string with the type of the data.
        
        :return: The Data Type.
        :rtype: str
        """
        return self.data_type
    
    def get_workflow_type(self):
        """Returns a string with the Worflow Type of the data.
        
        :return: The Workflow Type.
        :rtype: str
        """
        return self.workflow_type
    
    def get_legacy(self):
        """Returns a boolean specifying if the data will be downloaded
        from the GDC database or the GDC Legacy Archive.
        
        :return: The data source where the data will be downloaded.
        :rtype: bool
        """
        return self.legacy
    
    def get_access(self):
        """Returns a string with the type of access.
        
        :return: The type of access of the request.
        :rtype: str
        """
        return self.access
    
    def get_platform(self):
        """Returns a string the platform where data was produced
        
        :return:  platform where data was produced.
        :rtype: str
        """
        return self.platform
    
    def get_data_format(self):
        """Returns a string with the format of the data
        
        :return: The Format of the data.
        :rtype: str
        """
        return self.data_format
    
    def get_experimental_strategy(self):
        """Returns a string with the Experimental Strategy of the data.
        
        :return: The Experimental Strategy of the data.
        :rtype: str
        """
        return self.experimental_strategy
    
    def get_sample_type(self):
        """Returns a string with the Biospecimen source of the data.
        
        :return: The Type of the sample (Biospecimen source)
        :rtype: str
        """
        return self.sample_type

    def get_file_type(self):
        """Returns a string with the name of the project.
        
        :return: The name of the project.
        :rtype: str
        """
        return self.file_type

    def get_normalized(self):
        """Returns a boolean value that indicates if the data from
            the GDC Legacy Archive is normalized or not
        
        :return: Whether data is normalized or not.
        :rtype: bool
        """
        return self.normalized
    

    """
    SET
    """

    def set_project(self, project):
        """Set project's name value.
        
        :param project: Name of the project.
        :type project: str 
        """
        self.project = project
        
    def set_data_category(self, data_category):
        """Set Data Category value.
        
        :param data_category: Type of Data Category to download.
        :type data_category: str 
        """
        self.data_category = data_category
        
    def set_data_type(self, data_type):
        """Set Data Type value.
        
        :param data_type: Type of the data to download.
        :type data_type: str 
        """
        self.data_type = data_type
        
    def set_workflow_type(self, workflow_type):
        """Set Workflow Type value.
        
        :param workflow_type: Workflw Type to download.
        :type workflow_type: str 
        """
        self.workflow_type = workflow_type
        
    def set_legacy(self, legacy):
        """Set Legacy's value.
        
        :param legacy: Legacy boolean value.
        :type legacy: bool 
        """
        self.legacy = legacy
        
    def set_platform(self, platform):
        """Set the platform.
        
        :param platform: platform where data was produced
        :type platform: str 
        """
        self.platform = platform
        
    def set_data_format(self, data_format):
        """Set Data Format value.
        
        :param data_format: Format of the data to download.
        :type data_format: str 
        """
        self.data_format = data_format
        
    def set_experimental_strategy(self, experimental_strategy):
        """Set the Experimental Strategy value.
        
        :param experimental_strategy: Tye of Experimental Strategy to download.
        :type experimental_strategy: str 
        """
        self.experimental_strategy = experimental_strategy

    def set_sample_type(self, sample_type):
        """Set Sample Type value.
        
        :param sample_type: Type of the biospecimen's source to download.
        :type sample_type: str 
        """
        self.sample_type = sample_type
        
    def set_file_type(self, file_type):
        """Set File Type value.
        
        :param file_type: Set if data is normalized or not to download from
            GDC Legacy Archive.
        :type file_type: bool 
        """
        self.file_type = file_type



class DataProject():
    """This is a conceptual class representation a single downloaded project.
    Essential information such as the project's name, the path where it has been downloaded,
    the list of related files and the total size of the project is saved.
    
    :param project_name: Name of the projects. Defaults to "" (None).
    :type project_name: str
    :param project_name: Directory of the path where the project has been downloaded.
        Defaults to "" (None - Current directory).
    :type project_name: str  
    :param project_name: List of FileProject objects regarding the downloaded files.
        Deafults to [] (empty list).
    :type project_name: list[FileProject]  
    :param project_name: Total size of the project in MBs. Defaults to '0.0'.
    :type project_name: int  
    """

    def __init__(self):
        
        """Constructor method
        """          
        self.project_name = ""
        self.dir_path = "" # For example, TCGA/
        self.files = [] # List of FileProject objects
        self.size_download = 0.0


    """
    GET
    """

    def get_project_name(self):
        """Returns a string with the name of the project.
        
        :return: The name of the project.
        :rtype: str
        """
        return self.project_name

    def get_dir_path(self):
        """Returns a string with directory of the path where the project
            has been downloaded.
        
        :return: The directory path of the project.
        :rtype: str
        """
        return self.dir_path

    def get_files(self):
        """Returns a list of FileProject objects regarding the downloaded files.
        
        :return: List of FileProject objects.
        :rtype: list
        """
        return self.files

    def get_size_download(self):
        """Returns the total size of the project in MBs.
        
        :return: The size of the project in MBs.
        :rtype: int
        """
        return self.size_download

    def get_total_files(self):
        """Returns a integer with the number of the files downloaded.
        
        :return: The number of files downloaded.
        :rtype: int
        """
        return len(self.files)

    
    """
    SET
    """

    def set_project_name(self, project_name):
        """Set the project's name value.
        
        :param project_name: Name of the project.  
            Match with the ID of the cancer downloaded.
        :type project_name: str 
        """
        self.project_name = project_name

    def set_dir_path(self, dir_path):
        """Set the directory path where the project has been downloaded.
        
        :param dir_path: Directory path of the project.
        :type dir_path: str 
        """
        self.dir_path = dir_path

    def set_files(self, files):
        """Update theFileProject objects' list.
        
        :param files: list of downloaded files
        :type files: list 
        """
        self.files = files        
    
    def plus_size_download(self, mbs):
        """Set the size of the project calculating the
            total sum of all downloaded files in MBs.
        
        :param mbs: MBs of each file already downloaded.
        :type mbs: int 
        """
        self.size_download = self.size_download + mbs

    
    def to_string(self):
        """Print info about the project.
        """
        
        print("Data Project")
        print("\tProject Name: ", self.get_project_name())
        print("\tDirectory Path: ", self.get_dir_path())
        print("\tTotal Files: ", len(self.get_files()))
        print()

    def add_file(self, fObject):
        """Add a new FileObject to the FileObject objects' list.
        
        :param fObject: FileObject to add.
        :type: FileObject
        """
        self.files.append(fObject)

    def formatted_path(self, fObject):
        """Returns the full path where a file is located.
        
        :param fObject: the FileObject to format its path.
        :type fObject: FileObject

        :return: the full path where fObject is saved. 
            It is composed of the directory path where the project was downloaded,
            the id of fObject and the File Name of fObject.
        :rtype: str
        """
        
        return self.dir_path + "/" + fObject.get_id() + "/" + fObject.get_file_name()
    


class FileProject():
    """This is a conceptual class representation of the information and features
    of a downloaded file. Only the id_ will be a mandatory field due to the differences
    in the information that different data sources can provide.
    
    :param id_: UUID of the file. E.g: 48d05650-971e-4407-9cf1-7e9d71b1ff20
    :type id_: str  
    :param data_format: Format of the Data File, defaults to None.
    :type data_format: str, optional
    :param access: Indicator of whether acces to the data file is open or controlled,
        defaults to None.
    :type access: str, optional
    :param file_name: Name of the file. 
        E.g: BLAIN_p_TCGA_282_304_b2_N_GenomeWideSNP_6_G08_1348614.grch38.seg.v2.txt, defaults to None.
    :type file_name: str, optional
    :param submitter_id: A unique key to identify the case. Submitter ID is different from the universally
        unique identifier (UUID). E.g: 7f4e8b00-d73f-479e-a8f7-30a8d7aee093_dr12_allcnv, defaults to None.
    :type submitter_id: str, optional  
    :param data_category: A high-level data file category. E.g "Raw Sequencing
        Data" or "Transcriptome Profiling", dafults to None
    :type data_category: str, optional   
    :param type_: Type of the File, defaults to None.
    :type type_: str, optional   
    :param file_size: Size of the file in MB, defaults to None.
    :type file_size: int, optional
    :param created_datetime: Datetime of when the file has been created.
    :type created_datetime: datetime, optional
    :param md5sum: hash code. MD5 Checksum is used to verify the integrity of files,
        as virtually any change to a file will cause its MD5 hash to change.
    :type md5sum: str, optional
    :param updated_datetime: Datetatime of when a file had its latest update.
    :type updated_datetime: datetime, optional
    :param data_type: Data file type. Data Type is more granular than Data Category.
        E.g: "Align Reads" or "Gene Expression Quantification", defaults to None.
    :type data_type: str, optional
    :param state: Current state of the file. E.g: "releassed", defaults to None.
    :type state: str, optional
    :param experimental_strategy: Experimental strategies used for molecular
        characterization of the cancer.
    :type experimental_strategy: str, optional
    :param version: Number of the current version of the file.
    :type version: int, float, optional
    :param data_release: Number of the data release.
    :type data_release: float, optional
    :param entity_id: Also defined as Barcode. Entity ID is the primary identifier of
        biospecimen data within the TCGA project. E.g: TCGA-OR-A5JL-01A-11D-A29H-01, defaults to None.
    :type entity_id: str, optional
    """

    def __init__(self, id_, data_format = None, access = None, file_name = None, submitter_id = None,
               data_category = None, type_ = None, file_size = None, created_datetime = None, md5sum = None,
               updated_datetime = None, data_type = None, state= None,  experimental_strategy = None, version = None,
               data_release = None, entity_id = None): 
        """Constructor method
        """  

        self.id_ = id_
        self.data_format = data_format
        self.access = access
        self.file_name = file_name
        self.submitter_id = submitter_id # uuid
        self.data_category = data_category
        self.type = type_ # format (TSV, CSV, TXT...)
        self.file_size = file_size
        self.created_datetime = created_datetime
        self.md5sum = md5sum
        self.updated_datetime = updated_datetime
        self.data_type = data_type
        self.state = state
        self.experimental_strategy = experimental_strategy
        self.version = version
        self.data_release = data_release
        self.entity_id = entity_id


    """
    GET
    """

    def get_id(self):
        """Returns a string with the UUID of the file.
        
        :return: The UUID of the file.
        :rtype: str
        """
        return self.id_

    def get_data_format(self):
        """Returns a string with the Format of the file.
        
        :return: The Format of the file.
        :rtype: str
        """
        return self.data_format

    def get_access(self):
        """Returns a string whether the access is open or controlled.
        
        :return: The type of Access of the file.
        :rtype: str
        """
        return self.access

    def get_file_name(self):
        """Returns a string with the name of the file.
        
        :return: The name of the file.
        :rtype: str
        """
        return self.file_name

    def get_submitter_id(self):
        """Returns a string with the unique key to identify the case.
        
        :return: The ID to identify the case of the file.
        :rtype: str
        """
        return self.submitter_id

    def get_data_category(self):
        """Returns a string with the Data Category of the file.
        
        :return: The Data Category of the file.
        :rtype: str
        """
        return self.data_category

    def get_type(self):
        """Returns a string with the Format of the file (txt, tsv...)
        
        :return: The format of the file.
        :rtype: str
        """
        return self.type

    def get_file_size(self):
        """Returns a integer with the size of the file in MB.
        
        :return: The size of the file in MB.
        :rtype: int
        """
        return self.file_size

    def get_created_datetime(self):
        """Returns the date-time of the creation of the file.
        
        :return: The datetime of the creation of the file.
        :rtype: datetime
        """
        return self.created_datetime

    def get_md5sum(self):
        """Returns a string with the MD5 Checksum hash code.
        
        :return: A MD5 Checksum hash code.
        :rtype: str
        """
        return self.md5sum

    def get_updated_datatime(self):
        """Returns date-time of the latest update of the file.
        
        :return: The datetime of the the latest update of the file.
        :rtype: datetime
        """
        return self.updated_datetime

    def get_data_type(self):
        """Returns a string with the type of the file
        
        :return: The Data Type of the file.
        :rtype: str
        """
        return self.data_type

    def get_state(self):
        """Returns a string with the state of the file.
        
        :return: The state of the file.
        :rtype: str
        """
        return self.state

    def get_experimental_strategy(self):
        """Returns a string with the Experimental Strategy of the file.
        
        :return: The Experimental Strategy of the file.
        :rtype: str
        """
        return self.experimental_strategy

    def get_version(self):
        """Returns an integer/float with the version of the file.
        
        :return: The current version of the file.
        :rtype: int, float.
        """
        return self.version

    def get_data_release(self):
        """Returns a float number with the data release.
        
        :return: The data release.
        :rtype: float
        """
        return self.data_release

    def get_entity_id(self):
        """Returns a string with the Entity ID/Barcode of a TCGA's file.
        
        :return: The Entity ID of a file
        :rtype: str
        """
        return self.entity_id


    """
    SET
    """

    def set_id(self, id_):
        """Set the file's id value.
        
        :param id_: UUID of the file.
        :type id_: str 
        """
        self.id_ = id_

    def set_data_format(self, data_format):
        """Set Data Format value.
        
        :param data_format: Data Format of the file.
        :type data_format: str 

        """
        self.data_format = data_format

    def set_access(self, access):
        """Set the access type of the file, indicating if it is a
        open or controlled file.
        
        :param access: Type of access of the file.
        :type access: str 
        """
        self.access = access

    def set_file_name(self, file_name):
        """Set the file's name value.
        
        :param file_name: Name of the file downloaded.
        :type file_name: str 
        """
        self.file_name = file_name

    def set_submitter_id(self, submitter_id):
        """Set the unique key to identify the case.
        
        :param submitter_id: The ID to identify the case.
        :type submitter_id: str 
        """
        self.submitter_id = submitter_id

    def set_data_category(self, data_category):
        """Set Data Category value.
        
        :param data_category: Data Category of the file.
        :type data_category: str 
        """
        self.data_category = data_category

    def set_type(self, type_):
        """Set the Format of the file (txt, tsv...)
        
        :param type_: Format of the file.
        :type type_: str 
        """
        self.type = type_

    def set_file_size(self, file_size):
        """Set the file size in MBs.
        
        :param file_size: Size of the file in MB.
        :type file_size: int 
        """
        self.file_size = file_size

    def set_created_datetime(self, created_datetime):
        """Set the datetime of the creation of the file
        
        :param created_datetime: Creation's datetime.
        :type created_datetime: datetime 
        """
        self.created_datetime = created_datetime

    def set_md5sum(self, md5sum):
        """Set the MD5 Checksum hash code.
        
        :param md5sum: MD5 Checksum hash code.
        :type md5sum: str 
        """
        self.md5sum = md5sum

    def set_updated_datatime(self, updated_datetime):
        """Set the last update's datetame
        
        :param updated_datetime: Latest update's datetime'
        :type updated_datetime: datetime 
        """
        self.updated_datetime = updated_datetime

    def set_data_type(self, data_type):
        """Set the Type of the file.
        
        :param type_: Data Type of the file.
        :type type_: str 
        """
        self.data_type = data_type

    def set_state(self, state):
        """Set the state of the file.
        
        :param state: status of the file.
        :type state: str 
        """
        self.state = state

    def set_experimental_strategy(self, experimental_strategy):
        """Set the Experimental Strategy value.
        
        :param experimental_strategy: Tye of Experimental Strategy of the file.
        :type experimental_strategy: str 
        """
        self.experimental_strategy = experimental_strategy

    def set_version(self, version):
        """Set the version of the file.
        
        :param version: Latest version of the file.
        :type version: int, float 
        """
        self.version = version

    def set_data_release(self, data_release):
        """Set a number for the released data.
        
        :param data_release: Data release indicator of the file
        :type data_release: float 
        """
        self.data_release = data_release
    
    def set_entity_id(self, entity_id):
        """Set barcode's ID for a TCGA file.
        
        :param entity_id: Entity ID / Barcode of the file.
        :type entity_id: str 
        """
        self.entity_id = entity_id

    def to_string(self):
        """Print info about the file.
        """
        
        print("File")
        print("\tID: ", self.get_id())
        print("\tData Format: ", self.get_data_format())
        print("\tAccess: ", self.get_access())
        print("\tFile Name: ", self.get_file_name())
        print("\tSubmitter ID: ", self.get_submitter_id())
        print("\tData Category: ", self.get_data_category())
        print("\tType: ", self.get_type())
        print("\tFile Size: ", self.get_file_size())
        print("\tCreated Datetime: ", self.get_created_datetime())
        print("\tMd5sum: ", self.get_md5sum())
        print("\tUpdated Datetime: ", self.get_updated_datatime())
        print("\tData Type: ", self.get_data_type())
        print("\tState: ", self.get_state())
        print("\tExperimental Strategy: ", self.get_experimental_strategy())
        print("\tVersion: ", self.get_version())
        print("\tData Release: ", self.get_data_release())
        print("\tEntity ID: ", self.get_entity_id())
        print()
 

class Workflow(Task):
    """Task that represents the workflow of a program.
    The methods executed over the data can be saved and applied over bulk data 
    to replicate/expand the same analysis in the future.
    
    :param tasks: List of Task objects that creates a pipeline/workflow of Tasks.
        Defaults to [] (empty list).
    :type workflow: list[Log]  
    """

    """Constructor method
    """  
    def __init__(self):
        self.tasks = []

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = {}
        parameters['tasks'] = self.tasks
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.tasks = parameters['tasks']

    """
    GET
    """
        
    def get_tasks(self):
        """Returns the list of Task objects that represent a pipeline/workflow.
        
        :return: the list of Task objects that represent a pipeline/workflow.
        :rtype: list[Task].
        """
        return self.tasks
    
    def get_len_workflow(self):
        """Returns the number of Task objects that are in the current workflow.
        
        :return: Number of Task objects that are in the current workflow.
        :rtype: int
        """
        return len(self.tasks)
    
    def add_function(self, method, name = "", descp = ""):
        """Add a Task to the Workflow.
        
        :param method: Task to add in the workflow.
        :type method: Task 
        :param method: Title/Name of the Task. Defaults to "" (None).
        :type method: str 
        :param method: Description to explain what the Task does. 
            Defaults to "" (None).
        :type method: str 
        """
        
        self.get_tasks().append(method)


    def apply(self, obj):
        """Sequentially apply all Log Objects (Tasks) in the Workflow to a given object.
            It could be, for example, a DataObject or a DataFrame.
        
        :param obj: Object to apply the Workflow to.
        :type obj: DataObject, DataFrame

        :return: The object with all the Tasks applied.
        :rtype: DataObject, DataFrame
        """
        
        for pip in self.get_tasks():
            
            do = pip.apply(obj)
        
        return do
    
            
    def remove_last_function(self):
        """Removes from the Workflow the last function that was added to it.
        """
        
        (self.get_tasks()).pop()


    def generate_json(self, path = ""):
        """Generate an external file to save every Log object of Workflow

        :param path: file where dictionary generates will be saved
        :type: str
        """
        saved = "workflow.json"

        if path != "":
            saved = path

        if not exists(saved):
            try:
                open(saved, "w")
            except:
                print("Error. Unable to open or create {}".format(saved))

            try:
                list_task = []
                for l in self.get_tasks():
                    list_task.append(l.to_dict())
                
                with open(saved, "w+") as f_json:
                    json.dump(list_task, f_json)

            except:
                print("Error. Unable to write on file {}".format(saved))
        else:
            print("Error. File {} already exists".format(saved))


    """
    STR
    """

    def __str__(self):
        """Saves info about the pipeline/workflow.
        """
        
        for pip in self.get_tasks():
            print(pip)
            print("\n")
        
    def summary_workflow(self):
        """Print info about the pipeline/workflow.
        """
        print("\nWorkflow:")
                
        for pip in self.get_tasks():
            print(pip)
            print("\n")