# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""


import re
import os
import sys
import json
import copy
import requests
import numpy as np
import pandas as pd 
from threading import Thread



"""
######################################################

                    SERVER STATUS

######################################################
"""

def status_OK():
    """Message for 200 status code.
    """
    return "There is a connection to the server." # OK

def status_CREATED():
    """Message for 201 status code.
    """
    return "Connection created successfully." # CREATED

def status_UNDOCUMENTED():
    """Message for 400 status code.
    """
    return "The object does not exist." # UNDOCUMENTED (the object does not exist)

def status_UNAUTHORIZED():
    """Message for 401 status code.
    """
    return "Unauthorized access to the server." # UNAUTHORIZED
    
def status_FORBIDDEN():
    """Message for 403 status code.
    """
    return "Access denied to the server." # FORBIDDEN
    
def status_NOTFOUND():
    """Message for 404 status code.
    """
    return "Server not found." # NOT FOUND

def status_unknown():
    """Message for an unknown error.
    """
    return "Unknown error"


def status_code(stat_code):
    """Return the status response from the server
        
    :param stat_code: Current status of the checked URL endpoint.
    :type stat_code: int

    :return: Message explaining the obtained server's code.
    :rtype: str
    """

    return {
        200: status_OK(), 
        201: status_CREATED(),  
        400: status_UNDOCUMENTED(), 
        401: status_UNAUTHORIZED(), 
        403: status_FORBIDDEN(), 
        404: status_NOTFOUND(), 
    }.get(stat_code, status_unknown())



def check_server_status(endpt_status):
    """Check the status of the server.
        
    :param endpt_status: URL (endpoint) to be checked.
    :type endpt_status: str
    """
        
    url = requests.get(endpt_status)
        
    status = status_code(url.status_code)
    
    if(status != status_OK()):
        
        print("Error accessing the server. Try again later. ERROR: " + str(status)) 
        
        sys.exit(0)


def get_json_query(url_query, params_query = {}):
    """Get a structured JSON just by passing a URL, so it can be used by all the models.
    Each model will processes it as it is structured in its API.
    Params are empty by default, to get the data with or without parameters
    
    :param url_query: URL to do the request.
    :type url_query: str
    :param params_query: params to filter the query.
    :type params_query: dict

    :return: data obtained from the request.
    :rtype: json
    """
    
    url = requests.get(url_query, params = params_query)
    text = url.text
    data = json.loads(text)

    return data


def create_opin(filt):
    """Create a single in-operator in json format for the query, adding a field and its value.

    :param filt: values of a filter.
    :type filt: list[str]

    :return: the structured in-operator
    :rtype: dict
    """
    
    opin = {"op" : "in", 
            "content" : {
                "field" : filt[0], 
                "value" : filt[1]
                }
            }
    
    return opin


def find_parameter(value, param_results, param):
    """Search in the result's list the desired value.
        If it is not found, the method shows all available parameters.
        If missing is returned, that option will be removed from the result's list
        because it is an undesired value.

    :param value: desired value to find on a parameter.
    :type value: str
    :param param_results: available parameters.
    :type param_results: list[str]
    :param param: param where to find.
    :type param: str

    :return: param found.
    :rtype: str
    """
    
    if value.lower() not in param_results:
        print(str(((param.split(".")[-1]).replace("_", " ")).title()) + ": "  + str(value) + " was not found. Check available parameter: \n")
        
        if len(param_results) == 0 or (len(param_results) == 1 and "_missing" in param_results):
        
            print("\nNo data available for this field.")
            
        else:

            # param_results have at least one available parameter
            # remove _missing if it is an option
            try:
                param_results.remove("_missing")
            except:
                pass
        
            for pm in param_results:
                print("{:<2} ".format(pm.title()))
                
        sys.exit(0)
        
    else:
        
        return param_results.index(value.lower())


def print_json(json_obj):
    """Decodes a JSON object and display it in an elegant format per screen,

    :param json_obj: JSON object to print.
    :type json_obj: JSON Object
    """
    print(json.dumps(json_obj, indent=2, default = str)) # Display JSON with its correct identation



"""
######################################################

                    UTILITIES

######################################################
"""


def check_have_header(filename):
    """Check if the file has or not header.
        
    :param filename: Name of the file to check if has header or not.
    :type filename: str

    :return: 'True' if the file has header, 'False' if otherwise.
    :rtype: bool
    """

    try:
        with open(filename) as f:
            first = f.read(1)
    
    except:
        print("Unable to check header")

    return first not in '.-0123456789' #bytes


def check_arguments(arguments, metadata):
    """Check if the arguments of a function uses
    the expected type to receive.

    :param arguments: Arguments to check.
    :type arguments: list
    :param metadata: List with the type of each of the arguments.
    :type metadata: list
    """
    
    for inp in metadata.items():
        param = inp[0]
        typ = inp[1]["type"]

        # Check the parameter in the arguments
        arg_value = arguments[param]

        if not isinstance(arg_value, typ):
          print("{} type not match with {}".format(arg_value, typ))
          sys.exit(0)


def read_file(file, index = None, sep="\t"): 
    """Returns a DataFrame with the read file.

    :param file: File to read
    :type file: str
    :param index: Value of the index of the DataFrame.
    :type index: str
    :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
    :type sep: str

    :return: DataFrame with the file read.
    :rtype: DataFrame
    """
    
    try:
        # Check if the file has header or not
        hasHeader = check_have_header(file)
        
        if hasHeader:
            datos = pd.read_csv(file, sep=sep)
                    
        else:
            datos = pd.read_csv(file, sep=sep, header=None)
        
        if index != None:
            columns_datos = list(datos.columns.values)

            if index in columns_datos:
                datos = datos.set_index(index)
            
            else:
                print("{} is not a column of the file.".format(index))
                sys.exit(0)

        return datos   
    
    except:
        print("\nFile " + str(file) + " was not found.")
        sys.exit(0)
        

def copy_object(obj):
    """Comparison between objects doesn´t make a copy, 
    but just add another reference. 
    This method do a copy of an object.

    :param obj: Object to do the copy
    :type obj: Object

    :return: Copy of the object
    :rtype: object
    """
    
    return copy.deepcopy(obj)


def recursive_delete_data(dir_to_delete):
    """Search in the directory tree the input directory, so it remove its
        content recursively before remove the root directory.

    :param dir_to_delete: Directory path to delete its content.
    :type dir_to_delete: str
    """

    # Function next(os.walk(dirToDelete)) returns the following structure:
    #     [0] -> Root (directory)
    #     [1] -> Subdirectories
    #     [2] -> Files
    
    content = next(os.walk(dir_to_delete))
                
    for subd in content[1]:
        # The following path is the root + the subdirectory
        recursive_delete_data(content[0] + "/" + subd)
    
    for fl in content[2]:
        try:
            # Files need to indicate the root path 
            os.remove(content[0]+"/"+fl) # Files
            
        except OSError as err:
            
            print(err.strerror)
    
    try:
        
        os.rmdir(content[0]) # Remove the empty root at the end of the process
        
    except OSError as err:
        
        print(err.strerror)



def calc_plot_dim(n_elems):
    """Calculate the number of rows and columns of a given number
        in order to find the most squared and symetric image that it is possible

    :param n_elems: The number of elements.
    :type n_elems: int

    :return: The number of rows and columns.
    :rtype: int, int
    """
    
    n_rows = 0
    
    if n_elems == 1:
        
        n_rows = 1
        
    elif n_elems > 6:
        
        if n_elems == 7:
            
            n_rows = n_elems//4+1
            
        else:
            
            n_rows = n_elems//4
    else:
        
        n_rows = n_elems//2
        
    n_cols = int(np.ceil(n_elems / n_rows))
    
    lista_dimens = []
    
    while n_elems > 0:
        
        n_c = n_elems - n_cols
        
        if n_c >= 0:
            
            lista_dimens.append(n_cols)
            n_elems = n_elems - n_cols
            
        else:
            lista_dimens.append(n_elems)
            n_elems = -1


    return n_rows, n_cols



def thresh_by_perc(data, perc=10.0, axis = 0):
    """Calculate a limit by percentage by number of rows or columns

    :param data: the dataframe to calculate that limit according to number of rows or columns.
    :type data: DataFrame
    :param perc: portion of total elements (0.00 to 100.0).
    :type perc: float
    :param axis: Axis 0 means by rows, 1 by columns.
    :type axis: int

    :return: the mininum number of elements that consider a limit.
    :rtype: int
    """

    if isinstance(data, pd.DataFrame):

        if axis == 0:
            num_rows = len(data.index.values)
        
            return int(((100-perc)/100)*num_rows + 1)

        elif axis == 1: 
        
            num_columns = len(data.columns.values)

            return int(((100-perc)/100)*num_columns + 1)

        else:
            print("Axis only can be 0 or 1.")
            sys.exit(0)

    else:
        print("Data must be a DataFrame.")
        sys.exit(0)



# Print iterations progress while download the data
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def print_progress_bar(iteration, total, mb_iter, mb_total, prefix = 'Progress:', suffix = 'Complete', length = 50, decimals = 1,  fill = '█', printEnd = "\r"):
    """Print a bar to show the progress of the download.
    
    :param iteration: Iterator that indicates the current file that it is being downloaded.
    :type iteration: int 
    :param total: Total number of files to be downloaded.
    :type total: int 
    :param mb_iter: Size of the current file in MB.
    :type mb_iter: float 
    :param mb_total: Total size of the project in MB.
    :type mb_total: float 
    :param prefix: Message to appear next to the amount of current size downloaded.
    :type prefix: str 
    :param suffix: Message to appear next to the completed part.
    :type suffix: str 
    :param length: Total length of the progress bar. Defaults to 50.
    :type length: int 
    :param decimals: Integer value that indicated the percentage of downloaded data.
        Defaults to 1.
    :type decimals: int 
    :param fill: Charcater to print the progress of the bar. Defaults to '█'.
    :type fill: str 
    :param printEnd: Character to print at the end of the iteration.    
        Defaults to "\r" (carriage return). A carriage return means moving 
        the cursor to the beginning of the line.
    :type printEnd: str 
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + '-' * (length - filledLength)

    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({mb_iter} MB/{mb_total} MB downloaded)', end = printEnd)
    sys.stdout.write("\033[F")

    if iteration == total: 
        print()


# The class allows to use multithreading to join results
# The use of multiples threads will increase the speed reading the downloaded data
# https://www.bogotobogo.com/python/Multithread/python_multithreading_subclassing_creating_threads.php
       
class ThreadWithResult(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)



"""
######################################################

                  READ DATAOBJECT

######################################################
"""


def serialize(obj):
    """JSON serializer for objects not serializable by default json code
    """

    if isinstance(obj, pd.DataFrame):
        serial = obj.to_dict()
        return serial

    return obj.__dict__


# DataObject to Dictionary
def dict_from_dataobject(dobject):
    """Generates a Dictionary from DataObject information

    :param dobject: a object of class DataObject to extract information
    :type dobject: DataObject

    :return: a dictionary with the same information as input
    :rtype: dict
    """
    dictry = {
        "counts" : dobject.get_counts(),
        "dims" : dobject.get_counts_dims(),
        "obs" : dobject.get_obs(),
        "var" : dobject.get_var(),
        "logs" : dobject.get_log_pipeline(),
        "uns" : dobject.get_uns()

    }

    return dictry


# Dictionary to JSON
def json_from_dict(data):
    """Generate a Json from dictionary information

    :param data: a dictionary to extract information
    :type data: dict

    :return: a str with json syntax with the same information as input
    :rtype: str
    """

    try:
        # Change JSON Serializer class by default by an specific one
        # created by the developer
        return json.dumps(data, default=serialize)

    except:
        print("JSON could not be converted from Dictionary")
        sys.exit(0)


# JSON to Dictionary           
def dict_from_json(json_data):
    """Serialize a dictionary from json information

    :param json_data: a json structure to extract information
    :type json_data: str

    :return: a dictionary with the same information as input
    :rtype: dict
    """

    try:
        return json.loads(json_data, default=serialize)

    except:
        print("Dictionary could not be converted from JSON")
        sys.exit(0)


# JSON File to Dictionary
def dict_from_json_file(path):
    """Serialize a dictionary from json file

    :param path: a path with the json file to extract information
    :type path: str

    :return: a dictionary with the same information as input
    :rtype: dict
    """

    # Return if it is a file
    if os.path.isfile(path):

        with open("data_file.json", "r") as read_file:
            try:
                return json.load(read_file) # return a dict 
            except:
                print("JSON could not be converted to Dict.")
                sys.exit(0)


# Dictionary to JSON File
def json_file_from_dict(data, path = None):
    """Generate a Json file from dictionary information

    :param data: a dictionary to extract information
    :type data: dict
    :param path: path to save new file created. If None, it creates a file on the same
        path as program with the generic name 'object.json'
    :type path: dict
    """

    if path == None:
        dest = "object.json"
        
    else:    
        
        pattern = "^[\x00-\x7F]*(.txt|.csv|.tsv|.json|.dict|.dat|.data)$"
        
        if re.match(pattern, path):
            dest = path 
        else:
            tmp_path = path.split("/")
            tmp_path = tmp_path[:len(tmp_path) - 1]
            
            dest = "/".join(tmp_path) + "/object.json"

    with open(dest, "wb") as output_file:
        json.dump(data, output_file)