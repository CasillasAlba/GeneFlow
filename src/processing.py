# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

import re
import sys
import numpy as np
import pandas as pd
import scipy.stats as stat
from src import utils as ut
from abc import abstractmethod
from importlib import import_module


#from rnalysis import filtering

class Task():
    """This is a conceptual class representation of all Task on API development.
        All other project tasks inherit from it.
        The parameterization of a Task allows the instantiation of any child of Task with only its name.
        It is possible to generate the Reflection technique, and instantiation of the Child Task
        to be done from the base class (Task) without the need to duplicate code
        and organize all Task by the same pattern.

        
    """
    def __init__(self):
        pass
    
    @abstractmethod    
    def get_parameters(self):
        """Allow parameterization of a Task
        """
        pass 

    @abstractmethod    
    def set_parameters(self, parameters):
        """Will update the values of parameters of a Task
        """
        pass 

    # Ejecucion de las tareas

    @abstractmethod
    def apply(self):
        """Will be the main function of a Task to realize a specific job
        """
        pass

    # I/O    
    @staticmethod
    def instantiate(class_str):
        """Instantiate a class object of a Task by its path class

        :param class_str: the name of the class
        :type class_str: str

        :return: a instance of a Task
        :rtype: Task
        """

        # Class reflection
        try:
            module_path, class_name = class_str.rsplit('.', 1)
            module = import_module(module_path)
            klass = getattr(module, class_name)
        except (ImportError, AttributeError):
            raise ImportError(class_str)        
        # Dynamic instantiation
        instance = klass()
        return instance
    
    def to_dict(self):
        """Generate a dictionary from its parameters and generate a readable structure
            to replicate a Task Object later

        :return: a json object with the element of a Task organizes
        :rtype: str
        """

        json = {}
        json["type"] = self.__class__.__module__ + "." + self.__class__.__name__
        json.update( self.get_parameters() )
        return json 

    @staticmethod
    def from_dict(dictionary):
        """Read a dictionary (or json structure) and create a Object Task
            with all values which the previous object Task was saved.
            It is the method that allows replicate any child of Task only knowing this class

        :return: a replicate instance of the Task previosly saved on a json file or dictionary
        :rtype: Task
        """
        # Dynamic instantiation & parameterization
        instance = Task.instantiate(dictionary["type"])
        instance.set_parameters(dictionary)
        return instance

    def __str__ (self):
        """Allows stream a Task on output

        :return: a dictionary transform to readable str
        :rtype: str
        """
        return str(self.to_dict())



class DataTask(Task):
    """This is a conceptual class representation that organize
        all method associate with DataFrame as Task
    """
    def __init__(self):
        super().__init__()

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters={}
        return parameters
    

class ListTask(Task):
    """This is a conceptual class representation that organize
        all method associate with List or Aarray as Task
    """
    def __init__(self):
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        super().__init__()

    def get_parameters(self) -> dict:
        parameters={}
        return parameters



"""
######################################################

                OPERATIONS WITH DATA

######################################################
"""

# =============================================================================
#                               ACCESS
# =============================================================================


class DataColumnames(DataTask):
    """Returns list of the columns of the a DataFrame
    """
    def __init__(self):
        super().__init__()  

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)

    # Getters

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : ""
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "Name of columns"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # A list of the columns of the dataframe is returned
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A list of the columns of the dataframe
        :rtype: list
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):
            return list(data.columns.values)
        else:
            print("Input variable must be a dataframe.")
            sys.exit(0)
     
            
class DataRownames(DataTask):
    """Returns list of the rows of the a DataFrame
    """
    def __init__(self):
        super().__init__()  

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)

    # Getters

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : ""
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "Name of columns"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # A list of the rows of the dataframe is returned
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A list of the rows of the dataframe
        :rtype: list
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return list(data.index.values)



class DataExplainVariable(DataTask):
    """Given a DataFrame, return all information associated to a variable (row)
        
    :param var_id: identificator to select a specific variable
    :type var_id: str, int, float
    """
    def __init__(self, var_id):
        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.var_id = var_id

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['var_id'] = self.var_id
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.var_id = parameters['var_id']

    # Getters

    def get_var_id(self):
        """Returns the identificator of the specific variable

        :return: the identificadot
        :rtype: str, int, float
        """
        return self.var_id


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "var_id" : {
                    "type" : (str, int, float),
                    "description" : "ID of the variable to explain"
                }
            },
            "Apply Input" : {
                "samples" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame with information to explain"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (pd.Series, pd.DataFrame),
                    "description" : "Explained information about a variable"
                }
            }
        }
    

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # A list of the rows of the dataframe is returned
    def apply(self, samples):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a dataframe o series with information associated to this variable
        :rtype: DataFrame, Series
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(samples, pd.DataFrame):
            return DataSelectionName(self.get_var_id()).apply(samples)
        else:
            print("Input variable must be a dataframe.")
            sys.exit(0)



class DataExplainVariableColname(DataTask):
    """Given a DataFrame, return a certain information associated to a variable (row)
        
    :param var_id: identificator to select a specific variable
    :type var_id: str, int, float
    :param colnm: name of the column to extract that information
    :type colnm: str, int, float
    """
    def __init__(self, var_id, colnm):
        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.var_id = var_id
        self.colnm = colnm

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['var_id'] = self.var_id
        parameters['colnm'] = self.colnm
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.var_id = parameters['var_id']
        self.colnm = parameters['colnm']

    # Getters

    def get_var_id(self):
        """Returns the identificator of the specific variable

        :return: the identificadot
        :rtype: str, int, float
        """
        return self.var_id
    
    def get_colnm(self):
        """Returns the name of the column to extract certain information

        :return: name of the column
        :rtype: str, int, float
        """
        return self.colnm


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "var_id" : {
                    "type" : (str, int, float),
                    "description" : "ID of the variable to explain"
                },
                "colnm" : {
                    "type" : (str, int, float),
                    "description" : "Name of the column to extract variable's information"
                }
            },
            "Apply Input" : {
                "samples" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame with information to explain"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (pd.Series, pd.DataFrame),
                    "description" : "Explained information from column about a variable"
                }
            }
        }
    

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # A list of the rows of the dataframe is returned
    def apply(self, samples):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a dataframe o series with information associated to this variable and its column
        :rtype: DataFrame, Series
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(samples, pd.DataFrame):
            gen = DataExplainVariable(self.get_var_id()).apply(samples)

            return DataProjectionName(self.get_colnm()).apply(gen)

        else:
            print("Input variable must be a dataframe.")
            sys.exit(0)



class DataExplainAllVariableColname(DataTask):
    """Given a DataFrame, return a certain information of all variables
        
    :param colnm: name of the column to extract that information
    :type colnm: str, int, float
    """
    def __init__(self, colnm):
        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.colnm = colnm

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['colnm'] = self.colnm
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.colnm = parameters['colnm']

    # Getters
    
    def get_colnm(self):
        """Returns the name of the column to extract certain information

        :return: name of the column
        :rtype: str, int, float
        """
        return self.colnm


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "colnm" : {
                    "type" : (str, int, float),
                    "description" : "Name of the column to extract variable's information"
                }
            },
            "Apply Input" : {
                "samples" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame with information to explain"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (pd.Series, pd.DataFrame),
                    "description" : "Explained information from column about all variables"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, samples):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a dataframe o series with information associated to all variables and their column
        :rtype: DataFrame, Series
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(samples, pd.DataFrame):

            return DataProjectionName(self.get_colnm()).apply(samples)

        else:
            print("Input variable must be a dataframe.")
            sys.exit(0)



# =============================================================================
#                               CHECKING
# =============================================================================


class CheckElement(ListTask):
    """Check if a given element is on a list or array
        
    :param element: element to find on a list
    :type element: int, float, str, bool
    """
    def __init__(self, element = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.element = element

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['element'] = self.element
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.element = parameters['element']

    # Getters

    def get_element(self):
        """Returns the element to find on a list

        :return: a element to find
        :rtype: int, float, str, bool
        """
        return self.element


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "element" : {
                    "type" : (int, float, str, bool),
                    "description" : "element to find on a list"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a list to find a element"
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
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Check if a value is in the list
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The current list to be process
        :type list_: list
        :return: True if element is on a list, False otherwise
        :rtype: bool
        """

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
        
        
        
class CheckAllElement(ListTask):
    """Check if a given list of elements are on a list or array
        
    :param sublist: list of element to find
    :type sublist: list, ndarray
    """
    def __init__(self, sublist = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.sublist = sublist

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['sublist'] = self.sublist
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.sublist = parameters['sublist']

    # Getters

    def get_sublist(self):
        """Returns the list of element to find on a list

        :return: a list of element to find
        :rtype: list, ndarray
        """
        return self.sublist


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "sublist" : {
                    "type" : (list, np.ndarray),
                    "description" : "list with element to check"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a list to find a list of element"
                }
            },
            "Output" : {
                "return" : {
                    "type" : bool,
                    "description" : "True if all element is on list, False if not"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Return TRUE if all elements in sublist are in list_
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The current list to be process
        :type list_: list
        :return: True if all elements is on a list, False otherwise
        :rtype: bool
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        # Types of both list must coincide
        try:
            return all(elem in list_ for elem in self.get_sublist())
        
        except:
            print("Elements of the list extract have not the same type")
            sys.exit(0)
            
            
class CheckSubElement(ListTask):
    """It receives a list with elements where it is only needed to find 
        a SUBSTRING of the name to be valid.
        
    :param partial_sublist: list of element to find without need to match all name
    :type partial_sublist: list, ndarray
    """
    def __init__(self, partial_sublist = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.partial_sublist = partial_sublist

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['partial_sublist'] = self.partial_sublist
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.partial_sublist = parameters['partial_sublist']

    # Getters

    def get_partial_sublist(self):
        """Returns the list of element to find on a list

        :return: a list of element to find
        :rtype: list, ndarray
        """
        return self.partial_sublist


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "partial_sublist" : {
                    "type" : (list, np.ndarray),
                    "description" : "list with element to search"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a list to find a list of element"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "A list with the coincidence substring"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # It receives a list with elements where it is only needed to find 
    # a SUBSTRING of the name to be valid.
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The current list to be process
        :type list_: list
        :return: a list of coincidence with the large name
        :rtype: list, ndarray
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        # Avoid duplicated
        uniques_match = set()

        # Types of both list must coincide
        try:
            for elem in self.get_partial_sublist():
                # return a list of matches
                found = [value for value in list_ if elem in value] # add value to list if exists
                
                if found:
                    # Add a list to set but keep only uniques
                    uniques_match.update(found)
    
            return list(uniques_match)
    
        except:
            print("Elements of the list extract have not the same type")
            sys.exit(0)



class CheckSubElementShort(ListTask):
    """It receives a list with elements where it is only needed to find 
        a SUBSTRING of the name to be valid, but returns the short name founded
        
    :param partial_sublist: list of element to find without need to match all name
    :type partial_sublist: list, ndarray
    """
    def __init__(self, partial_sublist = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.partial_sublist = partial_sublist

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['partial_sublist'] = self.partial_sublist
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.partial_sublist = parameters['partial_sublist']

    # Getters

    def get_partial_sublist(self):
        """Returns the list of element to find on a list

        :return: a list of element to find
        :rtype: list, ndarray
        """
        return self.partial_sublist

    # Execution
    
    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "partial_sublist" : {
                    "type" : (list, np.ndarray),
                    "description" : "list with element to search"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a list to find a list of element"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "a list with coincidence almost partial"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The current list to be process
        :type list_: list
        :return: a list of coincidence with the short name
        :rtype: list, ndarray
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        uniques_match = dict()
        # Types of both list must coincide
        try:
            for elem in list_:
                # return a list of matches
                found = [value for value in self.get_partial_sublist() if value in elem] # add value to list if exists
                
                if found:
                    # Add a list to set but keep only uniques
                    uniques_match[elem] = found[0]
    
            return uniques_match
    
        except:
            print("Elements of the list extract have not the same type.")
            sys.exit(0)



# =============================================================================
#                               EXTRACTION
# =============================================================================


# extract_subdf_by_columns == Projection
# Projection means choosing which columns (or expressions) the query shall return.

class DataProjectionList(DataTask):
    """Extract a Sub-DataFrame selecting a specific column list.
        
    :param column_list: A list with the names of the columns to keep
    :type column_list: list
    """
    def __init__(self, column_list=[]):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.column_list = column_list

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['column_list'] = self.column_list
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.column_list = parameters['column_list']

    # Getters

    def get_column_list(self):
        """Returns the list with the column names to keep

        :return: a list with column names
        :rtype: list
        """
        return self.column_list


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "column_list" : {
                    "type" : (list, np.ndarray),
                    "description" : "list columns name to project"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with the selected columns in col_list
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrane
        :return: a Sub-DataFrame with selected columns
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):

            data_columns = DataColumnames().apply(data)
            
            inters = ListIntersection(self.get_column_list()).apply(data_columns)
            
            return data[inters]

        else:
            print("Input data must be a dataframe")
            sys.exit(0)



class DataProjectionIndex(DataTask):
    """Extract a Sub-DataFrame selecting a specific column by index.
        
    :param indx: a number of a column. Must be on range of lenght columns
    :type indx: int
    """
    def __init__(self, indx = 0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.indx = indx

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['indx'] = self.indx
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.indx = parameters['indx']

    # Getters

    def get_indx(self):
        """Returns the column index to project

        :return: a index column
        :rtype: int
        """
        return self.indx


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "indx" : {
                    "type" : int,
                    "description" : "index of the column"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific column"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by column index"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Search a column by index name
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected column
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):
        
            if self.get_indx() >= 0 and self.get_indx() < data.shape[1]:

                return data.iloc[:, self.get_indx()].to_frame()

            else:
                
                print("Value of column number exceed columns dimension")
                sys.exit(0)

        else:
            print("Input data must be a dataframe")
            sys.exit(0)



class DataProjectionName(DataTask):
    """Extract a Sub-DataFrame selecting a specific column by name.
        
    :param name: a name of a column. Must exists
    :type name: str
    """
    def __init__(self, name=""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.name = name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['name'] = self.name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.name = parameters['name']

    # Getters

    def get_name(self):
        """Returns the column name to project

        :return: a column name
        :rtype: str
        """
        return self.name


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "name" : {
                    "type" : str,
                    "description" : "name of the column to project"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by column name"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Search a column by column name
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected column
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):
            
            columnames = DataColumnames().apply(data)

            if self.get_name() in columnames:
        
                return data.loc[:, self.get_name()]

            else:
                
                print("Column name " + str(self.get_name()) + " does not exist in DataFrame")
                sys.exit(0)

        else:
            print("Input data must be a dataframe")
            sys.exit(0)



class DataProjectionRange(DataTask):
    """Extract a Sub-DataFrame selecting a range of columns
        
    :param ini_column: a initial position to project. Must be on range of lenght columns
    :type ini_column: int
    :param fini_column: a final position to project. Must be on range of lenght columns
    :type fini_column: int
    """
    def __init__(self, ini_column = 0, fini_column = -1):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.ini_column = ini_column
        self.fini_column = fini_column + 1 # + 1 for a closed interval

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['ini_column'] = self.ini_column
        parameters['fini_column'] = self.fini_column
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.ini_column = parameters['ini_column']
        self.fini_column = parameters['fini_column']

    # Getters

    def get_ini_column(self):
        """Returns the first column to be project

        :return: a initial position
        :rtype: int
        """
        return self.ini_column

    def get_fini_column(self):
        """Returns the last column to be project

        :return: a last position
        :rtype: int
        """
        return self.fini_row


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "ini_column" : {
                    "type" : int,
                    "description" : "initial value of column range"
                },
                "fini_column" : {
                    "type" : int,
                    "description" : "final value of column range"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by range"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with the selected columns in an interval
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with the selected columns in an interval
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):

            if (self.get_ini_column() >= 0 and self.get_ini_column() < data.shape[0]) and (self.get_fini_column() >= 0 and self.get_fini_column() <= data.shape[1]):

                if self.get_ini_column() < self.get_fini_column():
                    
                    return data.iloc[self.get_ini_column():self.get_fini_column()]

                else:
                    print("Init range cannot be equal or higher to final range.")
                    sys.exit(0)
            else:
                print("Column range exceed Dataframe dimensions.")
                sys.exit(0)
        else:
            print("Input data must be a dataframe")
            sys.exit(0)



class DataProjectionFilter(DataTask):
    """Extract a Sub-DataFrame selecting columns by filter.
        
    :param filter_: a filter to do the projection by columns
    :type filter_: pd.Series, pd.DataFrame
    """
    def __init__(self, filter_ = None):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.filter_ = filter_

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['filter_'] = self.filter_
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.filter_ = parameters['filter_']

    # Getters

    def get_filter(self):
        """Returns the filter to project DataFrame

        :return: a filter
        :rtype: pd.Series, pd.DataFrame
        """
        return self.filter_


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "filter_" : {
                    "type" : (pd.Series, pd.DataFrame),
                    "description" : "filter to do the projection by columns"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by filter"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with a filter
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame after apply a filter by columns
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(self.get_filter, pd.Series):
            
            return data.where(self.get_filter(), axis = 1)
        
        elif isinstance(self.get_filter(), pd.DataFrame):

            return data.loc[:, self.get_filter().columns]




class DataProjectionSubstring(DataTask):
    """Extract a Sub-DataFrame selecting a specific column list no matter if name doesn't match complete.
        
    :param column_list: A list with the names of the columns to keep
    :type column_list: list
    :param rename: check if should rename the column to the short name. True, then rename, False, not
    :type rename: bool
    """
    def __init__(self, column_list=[], rename = False):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.column_list = column_list
        self.rename = rename

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['column_list'] = self.column_list
        parameters['rename'] = self.rename
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.column_list = parameters['column_list']
        self.rename = parameters['rename']

    # Getters

    def get_column_list(self):
        """Returns the list with the column names (maybe shorter than columns DataFrame) to keep

        :return: a list with column names
        :rtype: list
        """
        return self.column_list

    def get_rename(self):
        """Returns if should rename the column to the short name

        :return: True if the columns of new DataFrame should rename, False otherwise
        :rtype: bool
        """
        return self.rename
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "column_list" : {
                    "type" : (list, np.ndarray),
                    "description" : "list columns name to project (can be shorted that column name of dataframe"
                },
                "rename" : {
                    "type" : bool,
                    "description" : "True if column should be renamed by short name, False if not"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to project specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by list and renamed or not"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Search in the column names (that have a long string) to check if there is a
    # valid substring in its value.
    # If rename == True -> rename the columns with the substring
    # E.g: For TCGA-E2-A1L7-11A-33R-A144-07 it will find TCGA-E2-A1L7   
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected columns
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        data_columns = DataColumnames().apply(data)
        
        coincidence = CheckSubElement(self.get_column_list()).apply(data_columns)

        if coincidence: # True if it is not empty
        
            if self.get_rename() == True:
                
                data = data[np.intersect1d(data_columns, coincidence)]
                
                dict_new_names = CheckSubElementShort(self.get_column_list()).apply(data_columns)

                return data.rename(columns=dict_new_names)
        
            return data[np.intersect1d(data_columns, coincidence)]
    
        return data



# Selection means which rows are to be returned.

class DataSelectionList(DataTask):
    """Extract a Sub-DataFrame selecting a specific row list.
        
    :param row_list: A list with the names of the rows to keep
    :type row_list: list
    """
    def __init__(self, row_list=[]):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.row_list = row_list

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['row_list'] = self.row_list
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.row_list = parameters['row_list']

    # Getters

    def get_row_list(self):
        """Returns the list with the row names to keep

        :return: a list with row names
        :rtype: list
        """
        return self.row_list


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "row_list" : {
                    "type" : (list, np.ndarray),
                    "description" : "list of rows name to select"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific rows"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe selected by list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with the selected rows in row_list
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected rows
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        data_rows = DataRownames().apply(data)
        
        inters = ListIntersection(self.get_row_list()).apply(data_rows)

        return data.loc[inters]



# def row_by_index(self, indx)
class DataSelectionIndex(DataTask):
    """Extract a Sub-DataFrame selecting a specific row by index.
        
    :param indx: a number of a row. Must be on range of lenght row
    :type indx: int
    """
    def __init__(self, indx = 0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.indx = indx

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['indx'] = self.indx
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.indx = parameters['indx']

    # Getters

    def get_indx(self):
        """Returns the row index to select

        :return: a row index
        :rtype: int
        """
        return self.indx


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "indx" : {
                    "type" : int,
                    "description" : "index of the row"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific row"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe selected by row index"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Search a row by index name
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected row
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):
        
            if self.get_indx() >= 0 and self.get_indx() < data.shape[0]:
            
                return data.iloc[self.get_indx()].to_frame().T # DataFrame

            else:
                
                print("Value of row number exceed rows dimension")
                sys.exit(0)

        else:
            print("Input data must be a dataframe")
            sys.exit(0)



class DataSelectionName(DataTask):
    """Extract a Sub-DataFrame selecting a specific row by name.
        
    :param name: a name of a row. Must exists
    :type name: str
    """
    def __init__(self, name = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.name = name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['name'] = self.name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.name = parameters['name']

    # Getters

    def get_name(self):
        """Returns the row name to select

        :return: a row name
        :rtype: str
        """
        return self.name


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "name" : {
                    "type" : str,
                    "description" : "name of the row to select"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe selected by row name"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Search a row by row name
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected row
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
            
        rownames = DataRownames().apply(data)

        if self.get_name() in rownames:
    
            return data.loc[self.get_name()].to_frame().T

        else:
            
            print("Row name " + str(self.get_name()) + " does not exist in DataFrame")
            sys.exit(0)




class DataSelectionRange(DataTask):
    """Extract a Sub-DataFrame selecting a range of rows
        
    :param ini_row: a initial position to select. Must be on range of lenght rows
    :type ini_row: int
    :param fini_row: a final position to select. Must be on range of lenght rows
    :type fini_row: int
    """
    def __init__(self, ini_row = 0, fini_row = -1):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.ini_row = ini_row
        self.fini_row = fini_row + 1 # + 1 for a closed interval

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['ini_row'] = self.ini_row
        parameters['fini_row'] = self.fini_row
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.ini_row = parameters['ini_row']
        self.fini_row = parameters['fini_row']

    # Getters

    def get_ini_row(self):
        """Returns the first row to be select

        :return: a initial position
        :rtype: int
        """
        return self.ini_row

    def get_fini_row(self):
        """Returns the last row to be select

        :return: a last position
        :rtype: int
        """
        return self.fini_row


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "ini_row" : {
                    "type" : int,
                    "description" : "initial value of row range"
                },
                "fini_row" : {
                    "type" : int,
                    "description" : "final value of row range"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific rows"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe selected by range"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with the selected rows in an interval
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with the selected rows in an interval
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        

        if (self.get_ini_row() >= 0 and self.get_ini_row() < data.shape[0]) and (self.get_fini_row() >= 0 and self.get_fini_row() <= data.shape[0]):

            if self.get_ini_row() < self.get_fini_row():
                
                return data.iloc[self.get_ini_row():self.get_fini_row()]

            else:
                print("Init range cannot be equal or higher to final range.")
                sys.exit(0)
        else:
            print("Row range exceed Dataframe dimensions.")
            sys.exit(0)



class DataSelectionFilter(DataTask):
    """Extract a Sub-DataFrame selecting rows by filter.
        
    :param filter_: a filter to do the selection by rows
    :type filter_: pd.Series, pd.DataFrame
    """
    def __init__(self, filter_ = None):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.filter_ = filter_

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['filter_'] = self.filter_
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.filter_ = parameters['filter_']

    # Getters

    def get_filter(self):
        """Returns the filter to select DataFrame

        :return: a filter
        :rtype: pd.Series, pd.DataFrame
        """
        return self.filter_


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "filter_" : {
                    "type" : (pd.Series, pd.DataFrame),
                    "description" : "filter to do the selection by rows"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific rows"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by filter"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with a filter
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame after apply a filter by rows
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(self.get_filter, pd.Series):
            
            return data.where(self.get_filter())
        
        elif isinstance(self.get_filter(), pd.DataFrame):

            return data.loc[self.get_filter().index]



class DataSelectionSeries(DataTask):
    """Extract a Sub-DataFrame selecting a specific row by series.
        
    :param series: A series with a structure to keep
    :type series: list
    """
    def __init__(self, series : pd.Series):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.series = series

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['series'] = self.series
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.series = parameters['series']

    # Getters

    def get_series(self):
        """Returns the series object with the rows to keep

        :return: a series with rows
        :rtype: Series
        """
        return self.series


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "series" : {
                    "type" : pd.Series,
                    "description" : "series to select"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to select specific rows"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe projected by series"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Create a subdataframe from a dataframe with the selected rows in series
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame with selected rows
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return data[self.get_series()]



# =============================================================================
#                               FILTERING
# =============================================================================


class FilterByIndex(DataTask):
    """Extract a Sub-DataFrame selected by a filter of index condition.
        For multi-index cases:
        index_cond must have index type (it is returned after  do dataframe.index)
        It keeps only the rows that are in index_cond
        
    :param index_cond: a condition as filter
    :type index_cond: str
    """
    def __init__(self, index_cond = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.index_cond = index_cond

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['index_cond'] = self.index_cond
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.index_cond = parameters['index_cond']

    # Getters

    def get_index_cond(self):
        """Returns the condition used to filter a DataFrame

        :return: a condition used
        :rtype: str
        """
        return self.index_cond
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "index_cond" : {
                    "type" : str,
                    "description" : "a condition to filter a dataframe"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to filter"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe filtered"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
                
    # For multi-index cases:
    # index_cond must have index type (it is returned after
    # do dataframe.index)
    # It keeps only the rows that are in index_cond
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrane to be process
        :type data: DataFrame
        :return: a Sub-DataFrame that keeps only the rows that are in index_cond
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        data = data[data.index.isin(self.get_index_cond())]

        return data



class FilterDictionary(DataTask):
    """Given a Dictionary, select only elements on a passed list
        
    :param list_var: new Count Matrix
    :type list_var: list
    """
    def __init__(self, list_var = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.list_var = list_var

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['list_var'] = self.list_var
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.list_var = parameters['list_var']

    # Getters

    def get_list(self):
        """Returns the list of elements to keep on dictionary

        :return: a list wit the elements selected
        :rtype: list
        """
        return self.list_var
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "list_var" : {
                    "type" : (list, np.ndarray),
                    "description" : "list of elements to filter a dictionary"
                }
            },
            "Apply Input" : {
                "dict_" : {
                    "type" : dict,
                    "description" : "a dictionary to filter"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dictionary filtered by list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, dict_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param dict_: The current Dictionary to be process
        :type dict_: dict
        :return: A new dictionary with filtered elements
        :rtype: dict
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        filter_dict = dict(filter(lambda k: k[0] in self.get_list(), dict_.items()))

        return filter_dict



# =============================================================================
#                             INTERSECTION
# =============================================================================


class ListIntersection(ListTask):
    """Obtains a new list with commom elements of two list or array
        
    :param sublist: secondary list to find commom elements
    :type sublist: list
    """
    def __init__(self, sublist = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.sublist = sublist

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['sublist'] = self.sublist
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.sublist = parameters['sublist']

    # Getters

    def get_sublist(self):
        """Returns the secondary list

        :return: a list with elements
        :rtype: list
        """
        return self.sublist


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "sublist" : {
                    "type" : (list, np.ndarray),
                    "description" : "second list to do a intersection"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a primary list"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "element that match in both list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Return a list which is the intersection between both input lists
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The primary list to be process
        :type list_: list
        :return: A list which is the intersection between both input lists
        :rtype: list
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        try:
            return [value for value in self.get_sublist() if value in list_]
        
        except:
            print("Elements of the list extract have not the same type.")
            sys.exit(0)



class ListIntersectionSubString(ListTask):
    """Obtains a new list with commom elements of two lists or array
        It is enough if only match partial name of element
        
    :param substrings: secondary list to find commom elements
    :type substrings: list
    """
    def __init__(self, substrings = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.substrings = substrings

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['substrings'] = self.substrings
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.substrings = parameters['substrings']

    # Getters

    def get_substrings(self):
        """Returns the secondary list

        :return: a list with elements
        :rtype: list
        """
        return self.substrings


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "substrings" : {
                    "type" : (list, np.ndarray),
                    "description" : "second list to do a intersection"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray),
                    "description" : "a primary list"
                }
            },
            "Output" : {
                "return" : {
                    "type" : (list, np.ndarray),
                    "description" : "element that match in both list at least part of the name of element in principal list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Find  common substring between lsubstrings and lstrings
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The primary list to be process
        :type list_: list
        :return: A list with founded common substring between two list
        :rtype: list
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if (isinstance(list_, list) and isinstance(self.get_substrings(), list)) or (isinstance(list_, np.ndarray) and isinstance(self.get_substrings(), np.ndarray)) :
            
            res = []
        
            for lit in self.get_substrings():
                
                # re.search() will search the regular expression pattern checking 
                # all lines of the input string. It will return a match object
                # when the pattern is found and "null" if the pattern is not found
                file_search = list(filter(lambda fl: re.search(lit, fl), list_))
                
                file_search = list(map(lambda x: x.replace(x, lit), file_search))
                
                res = res + file_search
            
            return res

        else:
            print("Both parameters must be lists")
            sys.exit(0)



# =============================================================================
#                                ZIP
# =============================================================================


class ListZip(ListTask):
    """Creates a new dictionary using one list
        for the keys, and other to its values
        
    :param sublist: a list with values of the dictionary
    :type sublist: list
    """
    def __init__(self, sublist = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.sublist = sublist

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['sublist'] = self.sublist
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.sublist = parameters['sublist']

    # Getters

    def get_sublist(self):
        """Returns the list that will be values of the new dictionary

        :return: a list with dictionary values
        :rtype: list
        """
        return self.sublist


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "sublist" : {
                    "type" : (list, np.ndarray, object),
                    "description" : "list of elements that will be values of dictionary"
                }
            },
            "Apply Input" : {
                "list_" : {
                    "type" : (list, np.ndarray, object),
                    "description" : "a list that will be keys of dictionary"
                }
            },
            "Output" : {
                "return" : {
                    "type" : dict,
                    "description" : "a dictionary created by both list"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Dictionary returned zipping two list with same length  
    def apply(self, list_):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param list_: The current list to be process
        :type list_: list
        :return: with the current list used as keys, creates and returns a dictionary with second list as values
        :rtype: dict
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if(len(list_) == len(self.get_sublist())):
            
            return dict(zip(list_, self.get_sublist()))

        else:
            print("Unable to zip the list. Check list's length.")
            sys.exit(0)



class DataZip(DataTask):
    """Creates a dictionary using information about two columns
        of a DataFrame
        
    :param key_column: column name of a DataFrame used for the dictionary keys
    :type key_column: str
    :param values_column: column name of a DataFrame used for the dictionary values
    :type values_column: str
    """
    def __init__(self, key_column = "", values_column = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.key_column = key_column
        self.values_column = values_column

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['key_column'] = self.key_column
        parameters['values_column'] = self.values_column
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.key_column = parameters['key_column']
        self.values_column = parameters['values_column']

    # Getters

    def get_key_column(self):
        """Returns the list that will be keys of the new dictionary

        :return: a list with dictionary values
        :rtype: list
        """
        return self.key_column

    def get_values_column(self):
        """Returns the list that will be values of the new dictionary

        :return: a list with dictionary values
        :rtype: list
        """
        return self.values_column
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "key_column" : {
                    "type" : str,
                    "description" : "name of the column of dataframe that will be values of dictionary"
                },
                "values_column" : {
                    "type" : str,
                    "description" : "name of the column of dataframe that will be keys of dictionary"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "the dataframe of columns will be extracted"
                }
            },
            "Output" : {
                "return" : {
                    "type" : dict,
                    "description" : "a dictionary created by columns of a dataframe"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: with both columns indicates, create and returns a dictionary
        :rtype: dict
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if(data.index.name != None):
        
            data = data.reset_index()
            
            if self.get_key_column() in data.columns:
            
                if self.get_values_column() in data.columns:
                    
                    zip_list_object = ListZip(data[self.get_values_column()])

                    return zip_list_object.apply(data[self.get_key_column()])
                
                else:
                    print("Column " + str(self.get_values_column()) + " is not a column of the dataframe.")
                    sys.exit(0)

            else:
                print("Column " + str(self.get_key_column()) + " is not a column of the dataframe.")
                sys.exit(0)



# =============================================================================
#                                RENAME
# =============================================================================

class RenameColumname(DataTask):
    """Rename all names of the columns triping the name
        
    :param pos_ini: position of the first character
    :type pos_ini: int
    :param pos_fin: position of the last character
    :type pos_fin: int
    """
    def __init__(self, pos_ini = 0, pos_fin = 0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.pos_ini = pos_ini
        self.pos_fin = pos_fin + 1

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['pos_ini'] = self.pos_ini
        parameters['pos_fin'] = self.pos_fin
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.pos_ini = parameters['pos_ini']
        self.pos_fin = parameters['pos_fin']

    # Getters

    def get_pos_ini(self):
        """Returns the initial position with the first character

        :return: initial position
        :rtype: int
        """
        return self.pos_ini

    def get_pos_fin(self):
        """Returns the last position with the first character

        :return: last position
        :rtype: int
        """
        return self.pos_fin
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "pos_ini" : {
                    "type" : int,
                    "description" : "initial position of name of the column"
                },
                "pos_fin" : {
                    "type" : int,
                    "description" : "final position of name of the column"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe that rename its columns"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with columns renamed selecting a portion of the original name"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    
    # Do a substring from the sample id (column name), selecting an initial and a final position
    # The input parameter "data" can be the dataframe or the list of the columns directly
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A DataFrame with the column renamed with a substring of the inner column name
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(data, pd.DataFrame):
            dflist = data.columns.to_list()
        
        elif isinstance(data, list):
            dfilst = data

    
        if isinstance(self.get_pos_ini(), int) and isinstance(self.get_pos_fin(), int):

            if self.get_pos_ini() < 0 or self.get_pos_fin() < 0:
    
                print("The position cannot be a negative number.")
                sys.exit(0)
    
            if self.get_pos_ini() > self.get_pos_fin():                        			
    
                print("Initial position cannot be higher than final position.")
                sys.exit(0)
    
            if self.get_pos_fin() > len(dflist[0]):
            
                print("Final position cannot be higher than the length of the column name.")
                sys.exit(0)
            
            if self.get_pos_ini() == self.get_pos_fin():
                print("Initial position cannot be the same as final position")
                sys.exit(0)

        else:
            print("Positions must be integer.")
            sys.exit(0)


        for i in range(0,len(dflist)):
    
            dflist[i] = dflist[i][self.get_pos_ini():self.get_pos_fin()]
            data.columns = dflist
    
        return data



class RenameIndex(DataTask):
    """Rename the name of the current index of DataFrame
        
    :param new_name: name that will replace current one
    :type new_name: str
    """
    def __init__(self, new_name = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)
        super().__init__()
        self.new_name = new_name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['new_name'] = self.new_name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.new_name = parameters['new_name']

    # Getters

    def get_new_name(self):
        """Returns the new name

        :return: the new name
        :rtype: str
        """
        return self.new_name
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "new_name" : {
                    "type" : str,
                    "description" : "new name of the dataframe index"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to rename its index"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with a renamed index"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: The same DataFrame with the index changed
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(self.get_new_name(), str):
             
            # We extract the name of the index, because it will be
            # the name of the column in result
            nm = data.index.names[0]
            
            # Remove the actual index
            data = data.reset_index()
            
            # The new column after remove the index is the prior index,
            # but now we have the possibility of rename it
            data = data.rename(columns={nm : self.get_new_name()})
            
            # We set it as the new index again
            data = data.set_index(self.get_new_name())
            
            return data
        else:
            print("The new name must be a string.")
            sys.exit(0)



# =============================================================================
#                                DUPLICATES
# =============================================================================


class DataDuplicates(DataTask):
    """Returns a DataFrame with the duplicates rows or columns of a input DataFrame
        
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    """
    def __init__(self, axis = 0, keep = False):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis # axis = 0 rows and axis = 1 columns
        self.keep = keep

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['axis'] = self.axis
        parameters['keep'] = self.keep
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.axis = parameters['axis']
        self.keep = parameters['keep']

    # Getters
    def get_axis(self):
        """Returns the axis selected to find duplicates

        :return: a number that could be 0 by rows and 1 by columns
        :rtype: int
        """
        return self.axis
    
    def get_keep(self):
        """Returns which duplicates will keep

        :return: could be "\"first\", \"last\" or False
        :rtype: str, bool
        """
        return self.keep
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "axis" : {
                    "type" : int,
                    "description" : "indicated 0 to row or 1 to columns"
                },
                "keep" : {
                    "type" : (str, bool),
                    "description" : "\"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to find duplicates"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe that only contains duplicated columns or rows"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe that only contains duplicated columns or rows
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        
        if isinstance(data, pd.DataFrame):
            
            if (self.get_axis() == 0 ) or (self.get_axis() == 1):

                try:
                    # duplicated function looks for duplicated columns and generates a boolean matrix
                    # If keep == "first" -> keeps only the first duplicated column
                    # If keep == "last" -> keeps only the last duplicated column
                    # If keep = False -> keeps both duplicated columns
                    
                    if self.get_axis() == 0:
                        
                        dup = data.index.duplicated(keep=self.get_keep())
                        
                        if True in dup: # If a True value is on the dup list, there is (minimun) a duplicated row
                            return data[dup]
                        
                        else:
                        
                            print("There aren't duplicated variables")
                            return pd.DataFrame()
                        
                    else:
                    
                        dup = data.columns.duplicated(keep = self.get_keep())
                        
                        if True in dup: # If a True value is on the dup list, there is (minimun) a duplicated column
                            
                            # All rows, but only the True Columns
                            return data.loc[:, dup]
                
                        else:
                            print("There aren't duplicated variables")
                            return pd.DataFrame()
        
                except:
                    print("Incorrect value of keep parameter. Options are: ")
                    print("\t\"first\", \"last\" or False")
                    sys.exit(0)
                    
            else:
                
                print("Incorrect parameter for axis. Options are: ")
                print("\t0 for rows, 1 for columns")
                sys.exit(0)
                        
        else:
            print("Data must be a DataFrame")
            sys.exit(0)



# =============================================================================
#                              INFORMATION
# =============================================================================


class Describe(DataTask):
    """Show a statistic description about a DataFrame
        
    :param perc: The percentiles to include in the output
    :type perc: list, float
    """
    def __init__(self, perc = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.perc = perc

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['perc'] = self.perc
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.perc = parameters['perc']

    # Getters

    def get_percentiles(self):
        """Returns the percentil or list of percentil

        :return: percentil or list of percentils selected
        :rtype: list, float
        """
        return self.perc
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "perc" : {
                    "type" : (list, np.ndarray, float),
                    "description" : "a float or list of float to indicates percentil"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to describe"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with result of describe"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Generate descriptive statistics that summarize the central tendency, dispersion and shape 
    # of the dataset’s distribution, excluding NaN values.
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: descriptive statistics that summarize the central tendency, dispersion and shape 
        of the dataset’s distribution, excluding NaN values
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return data.describe(percentiles=self.get_percentiles())



class CountTypes(DataTask):
    """Shows a Series containing counts of unique values of a specific column.
        It will be in descending order so that the first element is the most frequently-occurring element
        
    :param col_name: name of the column to count types
    :type col_name: str
    """
    def __init__(self, col_name = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_name = col_name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['col_name'] = self.col_name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_name = parameters['col_name']

    # Getters

    def get_columname(self):
        """Returns the name of the column to count

        :return: name of the column
        :rtype: str
        """
        return self.col_name
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "col_name" : {
                    "type" : str,
                    "description" : "name of the column to count its type"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to count types on a specific column"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "information about types of a column"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: information about types of a column
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if self.get_columname() in DataColumnames().apply(data):
            
            print(data[self.get_columname()].value_counts(),"\n")
            
        else:
            
            print(str(self.get_columname()) + "  is not a column name of data.")
            return -1
        

        
# =============================================================================
#                               REPLACE
# =============================================================================

class Replace(DataTask):
    """Replace DataFrame's selected values with another.

    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool
    """
    def __init__(self, to_replace = "Unknown", replaced_by = np.nan):
        
        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.to_replace = to_replace
        self.replaced_by = replaced_by

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['to_replace'] = self.to_replace
        parameters['replaced_by'] = self.replaced_by
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.to_replace = parameters['to_replace']
        self.replaced_by = parameters['replaced_by']

    # Getters

    def get_to_replace(self):
        """Returns the value to search to replace

        :return: a value to be replace
        :rtype: str, float, int, bool, list, ndarray
        """
        return self.to_replace
    
    def get_replaced_by(self):
        """Returns the new value

        :return: a value replace
        :rtype: str, float, int, bool
        """
        return self.replaced_by
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "to_replace" : {
                    "type" : (str, float, int, bool, list, np.ndarray),
                    "description" : "values to be replaced"
                },
                "replaced_by" : {
                    "type" : (str, float, int, bool),
                    "description" : "new value to replaced the value before"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to replace values"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with replaced values"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Replace all values by other value in a dataframe.
    # By default, it replaces all "Unknown"s by NAN,
    # but it is opened the possibility to let the user select both parameters.
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with all values indicates replaced by other value
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if isinstance(self.get_to_replace(), list):
            
            for replac in self.get_to_replace():
                data = data.replace(replac, self.get_replaced_by())
                
            return data
            
        else:
            return (data.replace(self.get_to_replace(), self.get_replaced_by()))



class FillNan(DataTask):
    """Replace nan values to mean of the row of DataFrame
        Only modifies values of DataFrame.
    """
    def __init__(self):
        super().__init__()

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)

    # Getters

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame to replace nan values to mean"
                }
            },
            "Output" : {
                "return" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame with nan replaced by mean"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())
    
    # Execution
    # Replace all nan values to mean
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A DataFrame with all nan values replace to mean
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        data_tmp = ut.copy_object(data)
       
        data_tmp = data_tmp.T.to_dict("list")
        
        data_tmp = ut.pd.DataFrame(data_tmp).T
        
        for i in data_tmp.columns[list(data_tmp.isna().sum()>0)]:
            data_tmp[i] = data_tmp[i].fillna((data_tmp[i].mean()))

        data_tmp.columns = DataColumnames().apply(data)

        return data_tmp



# =============================================================================
#                              TRANSPOSE
# =============================================================================


class Transpose(DataTask):
    """Do transpose to a DataFrame keeping the name of columns and index
        without multi-index
            
    :param index_name: the name of the index before transpose to keep it
    :type index_name: str, int
    """
    def __init__(self, index_name = ""):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.index_name = index_name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['index_name'] = self.index_name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.index_name = parameters['index_name']

    # Getters

    def get_index_name(self):
        """Returns the current index name before transpose

        :return: name of the current index
        :rtype: str, int
        """
        return self.index_name
    
    def set_index_name(self, new_name):
        self.index_name = new_name
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "index_name" : {
                    "type" : str,
                    "description" : "name of the index if exists"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to do a transpose"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a transpose of the dataframe"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Transpose of a dataframe taking into account the extra row
    # that it is created after the transpose: 
    # https://stackoverflow.com/questions/38148877/how-to-remove-the-extra-row-or-column-after-transpose-in-pandas
    # TO DO: tener en cuenta de que puede ser una numpy matriz!!
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with transpose of the input taking into account the extra row that it is created after the transpose
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        # If argument is empty, index_name will be the actual index of DataFrame
        if self.get_index_name() == "":
            
            self.set_index_name(data.index.names[0])
        
        # Remove the index
        mod_data = data.reset_index()
        
        # If after doing the transpose there is an index, a dataframe is generated
        # with the index (for the rows) which is impossible to remove!!!
        mod_transpose = mod_data.T
        
        # The first row will be the new index because the prior index (column 0)
        # it is now the row 0 (because of the transpose)
        # We keep it to change the column's names
        row0 = mod_transpose.iloc[0]
        
        # Row 0 is removed (selecting from row 1 to the rest of the rows)
        mod_transpose = mod_transpose[1:]
        
        # After remove the index, it will appear an index by defualt whose values
        # moves from 0 to n_rows. After the transpose, this numbers will be the column's names
        mod_transpose.columns = list(row0)
        
        mod_transpose = mod_transpose.reset_index()
        
        # The name of the first column will be changed for the new that there was initially (nm_index)
        # so we rename the name of the first column
        mod_data = mod_transpose.rename(columns={mod_transpose.columns[0] : self.get_index_name()})

        # We set the first column as the index
        mod_data = mod_data.set_index(self.get_index_name())
        
    
        return mod_data.astype(np.float)



# =============================================================================
#                                ADD
# =============================================================================

class DataAdd(DataTask):
    """This is a conceptual class representation that organize
        all method associate with add new element to DataFrame as Task
    """
    def __init__(self):
        super().__init__()
        
    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)



class AddColumn(DataAdd):
    """Add new column to a DataFrame
        
    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of DataFrame
    :type values: list, ndarray
    """
    def __init__(self, name_column = "", values = []):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.name_column = name_column
        self.values = values

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['name_column'] = self.name_column
        parameters['values'] = self.values
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.name_column = parameters['name_column']
        self.values = parameters['values']

    # Getters

    def get_name_column(self):
        """Returns the name of the new column

        :return: Name of the new column
        :rtype: str
        """
        return self.name_column

    def get_values(self):
        """Returns a list of values to initialize the new column

        :return: A list with the values of new column
        :rtype: list
        """
        return self.values
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "name_column" : {
                    "type" : str,
                    "description" : "name of the new column"
                },
                "values" : {
                    "type" : (list, np.ndarray),
                    "description" : "values of the new column. Must have the same lenght as number of rows"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "new column with its values will be added to this dataframe"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with the new column"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Add a new column with its values to a DataFrame
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: the input DataFrame with a new column with its values
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        # Values must have the same items as number of rows to proceed to add a column
        if len(self.get_values()) == data.shape[0]:

            data[self.get_name_column()] = self.get_values()

            return data
        
        else:
            print("Lenght of list of values must be the same as the row numbers of the DataFrame")
            
            sys.exit(0)



class AddLabel(DataAdd):
    """Add a special column to a DataFrame.
        It will be used on Machine Learning as the Target Column of Classification
    
    :param list_0s: List of values that consider as 0 on Label value
    :type list_0s: list, ndarray
    :param column_observed: Column of the DataFrame used to decide when value of target is 0 or 1
    :type column_observed: str
    :param name_label: values of the new column. Must have the same lenght as number of rows of DataFrame
    :type name_label: list, ndarray
    """
    def __init__(self, list_0s, column_observed, name_label = "label"):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.list_0s = list_0s
        self.column_observed = column_observed
        self.name_label = name_label

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['list_0s'] = self.list_0s
        parameters['column_observed'] = self.column_observed
        parameters['name_label'] = self.name_label
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.list_0s = parameters['list_0s']
        self.column_observed = parameters['column_observed']
        self.name_label = parameters['name_label']

    # Getters

    def get_list_0s(self):
        """Returns the list of values to consider as 0

        :return: List of values that consider as 0 on Label value
        :rtype: str
        """
        return self.list_0s

    def get_column_observed(self):
        """Returns the name of the target column

        :return: Name of the column to check to obtain label values
        :rtype: str
        """
        return self.column_observed

    def get_name_label(self):
        """Returns the name of the target column

        :return: Name of the new column as target
        :rtype: str
        """
        return self.name_label
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "list_0s" : {
                    "type" : (list, np.ndarray),
                    "description" : "list of values that consider as 0 on Label value"
                },
                "column_observed" : {
                    "type" : str,
                    "description" : "name of the column to check to obtain label values"
                },
                "name_label" : {
                    "type" : str,
                    "description" : "name of the new column where labels were saved"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "the dataframe with the column to calculate labels values"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with the new column of labels added"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Add a new column that represents the labels of data to classify
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with a new column that represents the labels of data to classify
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        num_rows = len(DataRownames().apply(data))

        datas_new_column = AddColumn(self.get_name_label(), ut.np.ones(num_rows, dtype=int)).apply(data)

        mask_pathologic = datas_new_column[self.get_column_observed()].str.lower().isin(self.get_list_0s())

        datas_new_column.loc[mask_pathologic, self.get_name_label()] = 0

        return datas_new_column



# =============================================================================
#                                DELETE
# =============================================================================


class DataDrop(DataTask):
    """This is a conceptual class representation that organize
        all method associate with Drop elements or values of a DataFrame as Task
    """
    def __init__(self):
        super().__init__()
        
    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)



class DropDuplicates(DataDrop):
    """Drop data duplicates of a DataFrame
        
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool
    """
    def __init__(self, axis = 0, keep = False, by_name = False):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis # axis = 0 rows and axis = 1 columns
        self.keep = keep
        self.by_name = by_name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['axis'] = self.axis
        parameters['keep'] = self.keep
        parameters['by_name'] = self.by_name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.axis = parameters['axis']
        self.keep = parameters['keep']
        self.by_name = parameters['by_name']

    # Getters
    def get_axis(self):
        """Returns the axis selected to drop duplicates

        :return: a number that could be 0 by rows and 1 by columns
        :rtype: int
        """
        return self.axis

    def get_keep(self):
        """Returns which duplicates will keep

        :return: could be "\"first\", \"last\" or False
        :rtype: str, bool
        """
        return self.keep
    
    def get_by_name(self):
        """Returns which kind of evaluation will do to know if a row or column is duplicate or not

        :return: True if only search by row or column name, False if analyze value of row or column
        :rtype: bool
        """
        return self.by_name
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "axis" : {
                    "type" : int,
                    "description" : "indicated 0 to row or 1 to columns"
                },
                "keep" : {
                    "type" : (str, bool),
                    "description" : "\"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows"
                },
                "by_name" : {
                    "type" : bool,
                    "description" : "True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if it is duplicated or not"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to drop duplicates rows or columns"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a new dataframe without duplicates rows or columns"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A new dataframe without duplicates rows or columns
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)


        if (self.get_axis() == 0 ) or (self.get_axis() == 1):

            try:
                if self.get_axis() == 0:
                    
                    if self.get_by_name() == False:
                        
                        return data.drop_duplicates(keep = self.get_keep())
                    
                    else:
                        
                        dup_rows = data.index.duplicated(keep = self.get_keep())
                        
                        return data.loc[~dup_rows]

                else: # axis == 1 then, columns
                
                    if self.get_by_name() == False:

                        if self.get_keep() != False:
    
                            return data.loc[:,~data.T.duplicated(keep = self.get_keep())]
                
                        else:
                
                            return data.loc[:,~data.columns.duplicated()]
                    else:
                        
                        dup_cols = data.columns.duplicated(keep = self.get_keep())

                        return data.loc[:,~dup_cols]

            except:
                print("Incorrect value of keep parameter. Options are: ")
                print("\t\"first\", \"last\" or False")
                sys.exit(0)

        else:
            print("Incorrect parameter for axis. Options are: ")
            print("\t0 for rows, 1 for columns")
            sys.exit(0)



class DropValues(DataDrop):
    """Drop values of a DataFrame

    :param to_delete: value to be delete.
    :type to_delete: str, float, int, bool, list, ndarray, 
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param method: all to check all values are the same, any if a partial value match
    :type method: str
    """
    def __init__(self, to_delete = np.nan, axis = 0, method = "all"):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.to_delete = to_delete
        self.axis = axis
        self.method = method

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['to_delete'] = self.to_delete
        parameters['axis'] = self.axis
        parameters['method'] = self.method
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.to_delete = parameters['to_delete']
        self.axis = parameters['axis']
        self.method = parameters['method']

    # Getters

    def get_to_delete(self):
        """Returns the value to be delete

        :return: value to be delete
        :rtype: str, float, int, bool, list, ndarray
        """
        return self.to_delete
    
    def get_axis(self):
        """Returns the axis selected to search value

        :return: 0 to row and 1 to columns
        :rtype: int
        """
        return self.axis
    
    def get_method(self):
        """Returns the method to check if value should be delete

        :return: all to check all values are the same, any if a partial value match
        :rtype: str
        """
        return self.method
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "to_delete" : {
                    "type" : (str, float, int, bool, list, np.ndarray),
                    "description" : "value to be delete"
                },
                "axis" : {
                    "type" : int,
                    "description" : "Indicates 0 to row and 1 to columns"
                },
                "method" : {
                    "type" : str,
                    "description" : "all to check all values are the same, any if a partial value match"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to drop some values"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a new dataframe within this values"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A new dataframe within this values
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        try:
            
            if self.get_method() == "all":
                
                data = data[~data.isin([self.get_to_delete()]).all(self.get_axis())]
                
            elif self.get_method() == "any":
                
                data = data[~data.isin([self.get_to_delete()]).any(self.get_axis())]
            
            return data
        
        except:
            print("Parameter to delete not found.")
            sys.exit(0)



class DropNanByThresh(DataDrop):
    """Drop DataFrame's values limited by Threshold.

    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param thresh: threshold to evaluate element. If value less than threshold, that will be selected to drop
    :type thresh: float, int
    """
    def __init__(self, axis = 0, thresh = 0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis
        self.thresh = thresh

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['axis'] = self.axis
        parameters['thresh'] = self.thresh
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.axis = parameters['axis']
        self.thresh = parameters['thresh']

    # Getters
    
    def get_axis(self):
        """Returns the axis selected to search value

        :return: 0 to row and 1 to columns
        :rtype: int
        """
        return self.axis
    
    def get_thresh(self):
        """Returns the threshold to evaluate element

        :return: If value less than it, that will be selected to drop
        :rtype: float, int
        """
        return self.thresh
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "axis" : {
                    "type" : int,
                    "description" : "indicates 0 to rows or 1 to columns"
                },
                "thresh" : {
                    "type" : (float, int),
                    "description" : "threshold to evaluate element. If value less than threshold, that will be selected to drop"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to drop values"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with drop element checking a threshold"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with drop element checking a threshold
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        try:
            data = data.dropna(axis = self.get_axis(), thresh = self.get_thresh())
            
            return data
        
        except:
            print("Error with the parameters.")
            sys.exit(0)



class DropRowsByColname(DataDrop):
    """Drop DataFrame's rows searching a value on a specific column

    :param col_name: name of the column to search
    :type col_name: str
    :param to_delete: value to find on a column to decide with row delete is exists
    :type to_delete: str, float, int, bool, list, ndarray
    """
    def __init__(self, col_name = "", to_delete = np.nan):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_name = col_name
        self.to_delete = to_delete


    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['col_name'] = self.col_name
        parameters['to_delete'] = self.to_delete
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_name = parameters['col_name']
        self.to_delete = parameters['to_delete']

    # Getters
    
    def get_columname(self):
        """Returns name of the column to search values to delete

        :return: name of the specific column
        :rtype: str
        """
        return self.col_name

    def get_to_delete(self):
        """Returns value search and delete

        :return: value to search
        :rtype: str, float, int, bool, list, ndarray
        """
        return self.to_delete
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "col_name" : {
                    "type" : str,
                    "description" : "name of the column to search values to delete"
                },
                "to_delete" : {
                    "type" : (str, float, int, bool, list, np.ndarray),
                    "description" : "value to be search and delete"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to drop rows"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with dropped rows after search a value on a specific column"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with dropped rows search a value on a specific column
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if self.get_columname() in data.columns:
            
            if isinstance(self.get_to_delete(), list):
                
                for delet in self.get_to_delete():
                    
                    data = data.loc[data[self.get_columname()] != delet]
            else:
                data = data.loc[data[self.get_columname()] != self.get_to_delete()]

            return data
                    
        else:
            print( str(self.get_columname()) + " column does not exist in the dataframe")
            sys.exit(0)



"""
######################################################

                QUANTITATIVE ANALYSIS

######################################################
"""


class QuantAnalysis(DataTask):
    """This is a conceptual class representation that organize
        all method associate with Quantitative Analysis of a DataFrame as Task
    """
    def __init__(self):
        super().__init__()
        
    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)



# =============================================================================
#                              CORRELATION
# =============================================================================

class VarCorrelation(QuantAnalysis):
    """Calculates correlation between variables of a DataFrame
    """
    def __init__(self):
        super().__init__()

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters
    
    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to do a correlation"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with the correlation done"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # Creates a dataframe with the correlation between the variables
    # with values from 0.0 to 1.0, being 1.0 the highest value of correlation
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with the correlation between the variables
            with values from 0.0 to 1.0, being 1.0 the highest value of correlation
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return data.corr()



class RemoveCorrelation(QuantAnalysis):
    """Remove the elements that has high correlation
    """
    def __init__(self, thresh = 0.0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.thresh = thresh


    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['thresh'] = self.thresh
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.thresh = parameters['thresh']

    # Getters
    
    def get_thresh(self):
        """Returns the limit used to decide with elements remove
        
        :return: a threshold value
        :rtype: float, int
        """
        return self.thresh
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "thresh" : {
                    "type" : (float, int),
                    "description" : "threshold to select values higher than this"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to delete high correlation"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with high correlation drop"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with high correlation removed
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if self.get_thresh() >= 0.0 and self.get_thresh() <= 1.0:
            
            data_tmp = Transpose().apply(data)
            
            remove = []

            corr = VarCorrelation().apply(data_tmp)
        
            corr_high = abs(corr) > self.get_thresh()

            counter = 1
            
            for i in corr_high.columns:
                
                for k in range(counter, len(corr_high.columns)):
                    
                    j = corr_high.columns[k]
                       
                    try:
                        if (corr_high[i][j]==True) and (i != j):
                            
                            if j not in remove:
                                remove.append(i)
                    except:
                        pass
                                
                counter = counter + 1
        
            remove = list(set(remove))
        
            #delete the elements that has high correlation
            data = data_tmp.drop(remove, axis=1)

            return data
        
        else:
            sys.exit(0)



# =============================================================================
#                               VARIANCE
# =============================================================================


class Variance(QuantAnalysis):
    """Calculates the variance of a matrix
    """
    def __init__(self):
        super().__init__()

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to calculate its variance"
                }
            },
            "Output" : {
                "return" : {
                    "type": dict,
                    "description" : "a dictionary with the variance of each gene"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dictionary with the variance of each variable
        :rtype: dict
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return ListZip(data.var()).apply(data.var().index)



# =============================================================================
#                              SELECTION
# =============================================================================


class TopVariables(QuantAnalysis):
    """Calculates the best n variables of a DataFrame

    :param n_var: Number of variables selected as the best
    :type n_var: int
    """
    def __init__(self, n_var = 0):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.n_var = n_var


    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['n_var'] = self.n_var
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.n_var = parameters['n_var']

    # Getters
    
    def get_num_var(self):
        """Returns how many variables will be returned

        :return: Number of variables selected as the best
        :rtype: int
        """
        return self.n_var
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "n_var" : {
                    "type" : int,
                    "description" : "number of variables to save"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : (dict, pd.DataFrame),
                    "description" : "a data structure to search the best n_var variables to select"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with n_var rows that are the best"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with n_var rows that are the best
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        
        if isinstance(data, pd.DataFrame):
            
            # Comprobar que n_var no sea mayor al numero total de variables,
            # y que sea > que 0
            
            variables = data.var(axis=1)
        
            if self.get_num_var() > len(variables):
                return variables.sort_values(ascending=False)
        
            # Choose n_var more expressed in data
            select_var =  variables.sort_values(ascending=False)[0:self.get_num_var()]
        
            highest_vars = data.loc[select_var.index]
            
            return highest_vars 
        
        elif isinstance(data, dict):
            
            var_sort = sorted(data.items(), key=lambda d: d[1], reverse=True)

            # Find top 1000 genes
            top_var = [i for i, j in var_sort[0:1000]]
            
            return top_var
            
        else:
            print("Input data must be a dataframe or dictionary")
            sys.exit(0)



class RemoveLowReads(QuantAnalysis):
    """Calculates the best n variables of a DataFrame

    :param thresh: threshold that indicates the limit to remove feature less than it
    :type thresh: float
    :param at_least: Indicates the minimum match to pass the selection
    :type at_least: int
    """
    def __init__(self, thresh = 0, at_least = 2):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.thresh = thresh
        self.at_least = at_least


    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['thresh'] = self.thresh
        parameters['at_least'] = self.at_least
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.thresh = parameters['thresh']
        self.at_least = parameters['at_least']

    # Getters
    
    def get_thresh(self):
        """Returns the threshold that limits to remove feature less than it

        :return: the Threshold
        :rtype: float
        """
        return self.thresh
    
    def get_at_least(self):
        """Returns the minimum match to pass the selection

        :return: the minimum match to pass
        :rtype: int
        """
        return self.at_least


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "thresh" : {
                    "type" : (float, int),
                    "description" : "threshold that indicates the limit to remove feature less than it"
                },
                "at_least" : {
                    "type" : int,
                    "description" : "Indicates the minimum match to pass the selection"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to remove features by threshold condition"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with less reads removed"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Execution
    # See from RNAlysis: https://guyteichman.github.io/RNAlysis/build/rnalysis.filtering.CountFilter.filter_low_reads.html
    # Remove features which have less then ‘threshold’ reads all columns
    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with removed features which have less then ‘threshold’ reads all columns
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        thresh_filter = data > self.get_thresh()

        # By Columns (Up to Down)
        thresh_at = thresh_filter.sum(axis=1) >= self.get_at_least()

        return data[thresh_at]



# =============================================================================
#                              NORMALIZATION
# =============================================================================


class Normalization(QuantAnalysis):
    """This is a conceptual class representation that organize
        all method associate with Normalization of a DataFrame as Task
    """
    def __init__(self):
        super().__init__()
        
    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)


class CPM(Normalization):
    """Normalize values of DataFrame using CPM method
        
    :param log_method: True if CPM will do with logaritmic alghoritm, False otherwise
    :type log_method: bool
    :param prior_count: If log is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size
    :type prior_count: int
    """
    
    def __init__(self, log_method = False, prior_count = 2):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.log_method = log_method
        self.prior_count = prior_count
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['log_method'] = self.log_method
        parameters['prior_count'] = self.prior_count
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.log_method = parameters['log_method'] 
        self.prior_count = parameters['prior_count'] 
    
    def get_log_method(self):
        """Returns if CPM will be apply with logaritmic or not

        :return: True if CPM will do with logaritmic alghoritm, False otherwise
        :rtype: bool
        """
        return self.log_method
    
    def get_prior_count(self):
        """Returns the ratio of a library size to the average library size

        :return: the ratio of a library size to the average library size
        :rtype: int
        """
        return self.prior_count
    

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "log_method" : {
                    "type" : bool,
                    "description" : "True if CPM will do with logaritmic alghoritm, False otherwise"
                },
                "prior_count" : {
                    "type" : int,
                    "description" : "If log_method is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe to normalize"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a normalized dataframe by one of two options"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    # Calculate the Counts per million (CPM) of the counts
    # RPM or CPM = (Number of reads mapped to gene x 10^6) / Total number of mapped reads    
    def cpm_(self, data):
        """Calculate the Counts per million (CPM) of the counts

        :param data: a DataFrame for which normalization is computed.
        :type data: DataFrame
        :return: a DataFrame with CPM computed
        :rtype: DataFrame
        """
        
        # The method should work both with a dataframe or a numpy array
        # https://www.w3schools.com/python/ref_func_isinstance.asp
        if isinstance(data, pd.DataFrame):
            # Gene counts must be non-negative
            assert data.to_numpy().min() >= 0.0  
        else:
            # Convert into numpy array
            data = np.asarray(data, dtype = np.float64)
            # Gene counts must be non-negative
            assert np.min(data) >= 0.0  
    
        data_sum = data.sum(axis=0)
    
        # Ignore warnings of division by 0
        with np.errstate(invalid="ignore"):
            cpm_value = 1e6 * data / data_sum
    
            # The samples with zeros for all genes get nan but should be 0.0
            np.nan_to_num(cpm_value, copy=False)
    
        return cpm_value
    
    # Calculate the Log CPM of the counts
    # https://bioinformatics.stackexchange.com/questions/4985/how-is-prior-count-used-by-edgers-cpm#:~:text=edgeR's%20cpm%20function%20has%20an,would%20be%20equal%20to%20prior.   
    def cpm_log(self, data):
        """Calculate the Log CPM of the counts

        :param data: a DataFrame for which normalization is computed.
        :type data: DataFrame
        :return: a DataFrame with CPM Log computed
        :rtype: DataFrame
        """

        # The method should work both with a dataframe or a numpy array
        # https://www.w3schools.com/python/ref_func_isinstance.asp
        if isinstance(data, pd.DataFrame):
            # Gene counts must be non-negative
            # data = data.astype(np.float)
            assert data.to_numpy().min() >= 0.0

        else:
            # Convert into numpy array
            data = np.asarray(data, dtype = np.float64)
            # Gene counts must be non-negative
            assert np.min(data) >= 0.0  
    
        # First, we need to calculate a library size
        lib_size = data.sum(axis = 0)
        
        # Calculate the average library size and the adjusted priors
        ave_lib = np.mean(lib_size)
        adjusted_prior = self.get_prior_count() * lib_size / ave_lib
        
        # Update the library sizes
        adjusted_lib_size = lib_size + 2 * adjusted_prior
        
        # Now we can compute the CPM
        customCPM = ((np.log(data + adjusted_prior) - np.log(adjusted_lib_size) + np.log(1000000)) / np.log(2))
        
        return customCPM 

    # Execution

    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame to computed one of the CPM method
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        if self.get_log_method(): 
            return self.cpm_log(data)

        return self.cpm_(data)




"""
######################################################

                    RNA PROCESS

######################################################
"""


class GeneAnalysis(QuantAnalysis):
    """This is a conceptual class representation that organize
        all method associate with Gene Analysis of a DataFrame as Task
    """
    def __init__(self):
        super().__init__()
        
    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection
        
        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)



class SampleCondition(GeneAnalysis):
    """Classify a given sample analyze its barcode

    :param sep: indicates with character is used to separate values
    :type sep: str
    """
    def __init__(self, sep = "-"):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.separator = sep
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['separator'] = self.separator

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.separator = parameters['separator'] 

    # Getter

    def get_separator(self):
        """Returns the character used to separate the name of the barcode

        :return: character to separate barcode on segment
        :rtype: str
        """
        return self.separator


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "sep" : {
                    "type" : str,
                    "description" : "indicates with character is used to separate values"
                }
            },
            "Apply Input" : {
                "brc_string" : {
                    "type" : str,
                    "description" : "a string that contains information about a tumor"
                }
            },
            "Output" : {
                "return" : {
                    "type": str,
                    "description" : "classification of this sample"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, brc_string):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param brc_string: The current str to be process
        :type brc_string: str
        :return: classification of the sample by its barcode passed on input
        :rtype: str
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        brc_string = self.get_data()
                
        # brc_example = "TCGA-02-0001-01C-01D-0182-01"
        #                  0  1   2   3    4   5   6
        brc_sample__vial = 3 #num-letra => 01C ; num(sample) letra(vial)
    
        brc_splited = brc_string.split(sep = self.get_data())
    
        sample_vial = brc_splited[brc_sample__vial] # String, array de char ["0", "1", "C"]
    
        sample_ = sample_vial[:2] 
        #vial_ = sample_vial[-1] # String, array de char ["0", "1", "C"]
    
        try:
            sample_type = int(sample_)
    
            if sample_type >= 1 and sample_type <= 9:
                return "TUMR" # tumor sample
            
            elif sample_type >= 10 and sample_type <= 19:
                return "NORM" # normal sample
            
            elif sample_type >= 20 and sample_type <= 29:
                return "CTRL" # control sample
    
            else:
                print("Unable to stablished the type of sample")
                return "NAN"
    
        except:
            print("Unable to stablished the type of sample")
            return "NAN"



class StatTest(GeneAnalysis):
    """Generate a DataFrame with several statistic information like FoldChange, T, P-Value and a DiffExpr

    :param grouped_by: name of the column of clinical data to organize data
    :type grouped_by: str
    :param group_1: one of the type to create a group
    :type group_1: str
    :param group_2: another type to create a group
    :type group_2: str
    :param clinical_data: a dataframe with clinical information of the data
    :type clinical_data: DataFrame
    """
    def __init__(self, grouped_by, group_1, group_2, clinical_data):

        arguments = locals()
        
        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.grouped_by = grouped_by
        self.group_1 = group_1
        self.group_2 = group_2
        self.clinical_data = clinical_data
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['grouped_by'] = self.grouped_by
        parameters['group_1'] = self.group_1
        parameters['group_2'] = self.group_2
        parameters['clinical_data'] = self.clinical_data

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.grouped_by = parameters['grouped_by']
        self.group_1 = parameters['group_1']
        self.group_2 = parameters['group_2']
        self.clinical_data = parameters['clinical_data']

    # Getter
    
    def get_grouped_by(self):
        """Returns name of the column of clinical data to organize data

        :return: name of the column to organize
        :rtype: str
        """
        return self.grouped_by

    def get_group_1(self):
        """Returns one of the type to create a group

        :return: one of the type
        :rtype: str
        """
        return self.group_1

    def get_group_2(self):
        """Returns another type to create a group

        :return: another type
        :rtype: str
        """
        return self.group_2

    def get_clinical_data(self):
        """Returns a dataframe with clinical information of the data

        :return: a DataFrame with clinical data
        :rtype: DataFrame
        """
        return self.clinical_data


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "grouped_by" : {
                    "type" : str,
                    "description" : "name of the column of clinical data to organize data"
                },
                "group_1" : {
                    "type" : str,
                    "description" : "one of the type to create a group"
                },
                "group_2" : {
                    "type" : str,
                    "description" : "another type to create a group"
                },
                "clinical_data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with clinical information of the data"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with the count matrix"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a dataframe with several stadistic information: FoldChange, T, P-Value and a DiffExpr"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: A dataframe with several stadistic information: FoldChange, T, P-Value and a DiffExpr
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
                
        # All Columns of Clinical Information
        columns_clinical = DataColumnames().apply(self.get_clinical_data())

        if self.get_grouped_by() in columns_clinical:
            
            # Get elements of group 1
            sample_by_group_1 = list(self.get_clinical_data()[self.get_clinical_data()[self.get_grouped_by()] == self.get_group_1()].index.values)

            # Get elements of group 2
            sample_by_group_2 = list(self.get_clinical_data()[self.get_clinical_data()[self.get_grouped_by()] == self.get_group_2()].index.values)

            # Get counts by condition of group 1
            proc_group_1 = DataProjectionList(sample_by_group_1).apply(data)

            # Get counts by condition of group 2
            proc_group_2 = DataProjectionList(sample_by_group_2).apply(data)


            if not proc_group_1.empty and not proc_group_2.empty:
                
                # Get means of group 1 of every row

                group_1_means = proc_group_1.mean(axis = 1)

                # Get means of group 2 of every row
                
                group_2_means = proc_group_2.mean(axis = 1)

                # Get foldchanges
                foldchanges = list(np.log2(np.divide(group_1_means, group_2_means)))

                # Get P-Value
                p_values = []
                t_test = []
                
                indx_name = data.index.name
                index_df = DataRownames().apply(data)

                for row in index_df:
                    # avoid warning divide by zero (they will be Zero)
                    with np.errstate(divide="ignore"):
                        ttest_result = stat.ttest_ind(proc_group_1.loc[row, :],  proc_group_2.loc[row, :])
                        p_values.append(ttest_result[1])
                        t_test.append(ttest_result[0])


                transformed_pvals = list(-1*np.log10(p_values))

                # Create a dataframe with data calculated
                pdict = {
                    indx_name : index_df
                }

                df_stat_test = pd.DataFrame(pdict)

                df_stat_test = df_stat_test.set_index(indx_name)

                df_stat_test = AddColumn("FoldChange", foldchanges).apply(df_stat_test)

                df_stat_test = AddColumn("T", t_test).apply(df_stat_test)

                df_stat_test = AddColumn("P-Value", transformed_pvals).apply(df_stat_test)

                df_stat_test = AddColumn("DiffExpr", np.zeros(df_stat_test.shape[0])).apply(df_stat_test)

                #assign each point on the plot a color based on its coordinates
                #points that are in the top right or top left should be highlighted 
                for index, row in df_stat_test.iterrows():
                    if row["P-Value"] > 2:
                    
                        if row["FoldChange"] > 0.5:
                            df_stat_test.loc[row.name, "DiffExpr"] = "Up"

                        elif row["FoldChange"] < -0.5:
                            df_stat_test.loc[row.name, "DiffExpr"] = "Down"
                        
                        else:
                            df_stat_test.loc[row.name, "DiffExpr"] = "Not Diff"
                        
                    else:
                        df_stat_test.loc[row.name, "DiffExpr"] = "Not Diff"
                
                return df_stat_test

            else:
                print("Error. Groups have not been formed correctly. Check groups information, ({}, {})".format(self.get_group_1(), self.get_group_2()))
                sys.exit(0)
        else:
            print("Error. ({}) is not a column of clinical information".format(self.get_grouped_by()))
            sys.exit(0)



class GenesLenght(GeneAnalysis):
    """Given a DataFrame with the genemodel information, return all lenght of the actual studied genes
        
    :param lenght_columname: name of the column that contains information about their lenght
    :type lenght_columname: str
    """
    def __init__(self, lenght_columname = "width"):
        self.lenght_columname = lenght_columname
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['lenght_columname'] = self.lenght_columname

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.lenght_columname = parameters['lenght_columname'] 

    # Getter

    def get_lenght_columname(self):
        """Returns name of the column of Genemodel with lenght information

        :return: name of the lenght column
        :rtype: str
        """
        return self.lenght_columname


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "lenght_columname" : {
                    "type" : str,
                    "description" : "Name of the column that represents Lenght Gene on model"
                }
            },
            "Apply Input" : {
                "genemodel" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe that represents genecode"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a DataFrame with lenghts of the genes"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, genemodel):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param genemodel: The current DataFrame to be process
        :type genemodel: DataFrame
        :return: a dataframe with lenghts of the genes
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return DataProjectionName(self.get_lenght_columname()).apply(genemodel)



class GeneIDLenght(GeneAnalysis):
    """Returns lenght of a specific gen indicates
        
    :param gen_id: identifier of the gene on a genemodel
    :type gen_id: str
    """
    def __init__(self, gen_id):
        self.gen_id = gen_id
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['gen_id'] = self.gen_id

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.gen_id = parameters['gen_id'] 

    # Getter

    def get_gen_id(self):
        """Returns name of the gen to obtain lenght information

        :return: name of the gen
        :rtype: str
        """
        return self.gen_id


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "gen_id" : {
                    "type" : str,
                    "description" : "Identifier of the Gene"
                }
            },
            "Apply Input" : {
                "genemodel" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe that represents genecode"
                }
            },
            "Output" : {
                "return" : {
                    "type": int,
                    "description" : "lenght of the gene"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, genemodel):
        """It is the main function of the class. It reproduces the
            corresponding task of the class for which it was created
        
        :param data: a DataFrame with genemodel information
        :type data: DataFrame
        :return: the lenght of a specific gen selected
        :rtype: int
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        return int(DataExplainVariableColname(self.get_gen_id(), "width").apply(genemodel).values)



class RPK(GeneAnalysis):
    """Normalize values of DataFrame using RPK method
        
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame
    """
    def __init__(self, genemodel):
        self.genemodel = genemodel
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['genemodel'] = self.genemodel

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.genemodel = parameters['genemodel'] 

    # Getter

    def get_genemodel(self):
        """Returns a DataFrame with genemodel information

        :return: Genemodel
        :rtype: DataFrame
        """
        return self.genemodel


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "genemodel" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame that contains information about Genes"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with Count Matrix (with its rows and columns)"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "RPK calculated on Count Matrix"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with computed RPK
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        genes_list = DataRownames().apply(data)

        count_matrix_rpk = ut.copy_object(data)
                        
        genes_lenght = GenesLenght().apply(self.get_genemodel())

        # In Kilobase
        genemodel_kilo = genes_lenght / 1000

        rpk = count_matrix_rpk.loc[genes_list] / genemodel_kilo.loc[genes_list].values

        return rpk



class TPM(GeneAnalysis):
    """Normalize values of DataFrame using TPM method
        
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame
    """
    def __init__(self, genemodel):
        self.genemodel = genemodel
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['genemodel'] = self.genemodel

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.genemodel = parameters['genemodel'] 

    # Getter

    def get_genemodel(self):
        """Returns a DataFrame with genemodel information

        :return: Genemodel
        :rtype: DataFrame
        """
        return self.genemodel


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "genemodel" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame that contains information about Genes"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with Count Matrix (with its rows and columns)"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "a DataFrame with count matrix normalized"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with computed TPM
        :rtype: DataFrame
        """
        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        """
        TPM NORMALIZATION
        Step 1 --> Normalize for gene length (kilobase) => Read per Kilobase (RPK)
        Step 2 --> Normalize for sequencing depth (divide by total RPK)
        """
        genes_list = DataRownames().apply(data)

        samples_list = DataColumnames().apply(data)

        # STEP 1

        # RPK Calculated

        rpk = RPK(self.get_genemodel()).apply(data)

        # Step 2

        tpm = np.zeros(shape=data.shape)
        rpk_values = rpk.values

        # Calculating TPM... Step 2:

        for i in range(data.shape[0]):
        
            rpk_depth = np.sum(rpk_values[i, :])

            scaled_rpk_depth = rpk_depth / 1000000

            # Divide by Zero avoid. Will be NaN
            with np.errstate(invalid="ignore"):  
                tpm[i, :] = rpk_values[i, :] / scaled_rpk_depth

                # Transform NaN to 0
                np.nan_to_num(tpm[i, :], copy=False)
        
        # Finish TPM
        tpm_dframe = pd.DataFrame(tpm, index=genes_list, columns=samples_list)
        tpm_dframe.index.name = data.index.name

        return tpm_dframe



class RPKM(GeneAnalysis):
    """Normalize values of DataFrame using RPKM method
        
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame
    """
    def __init__(self, genemodel):
        self.genemodel = genemodel
        
    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['genemodel'] = self.genemodel

        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.genemodel = parameters['genemodel'] 

    # Getter

    def get_genemodel(self):
        """Returns a DataFrame with genemodel information

        :return: Genemodel
        :rtype: DataFrame
        """
        return self.genemodel


    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters" : {
                "genemodel" : {
                    "type" : pd.DataFrame,
                    "description" : "a DataFrame that contains information about Genes"
                }
            },
            "Apply Input" : {
                "data" : {
                    "type" : pd.DataFrame,
                    "description" : "a dataframe with Count Matrix (with its rows and columns)"
                }
            },
            "Output" : {
                "return" : {
                    "type": pd.DataFrame,
                    "description" : "RPKM calculated on Count Matrix"
                }
            }
        }


    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())


    def apply(self, data):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created
        
        :param data: The current DataFrame to be process
        :type data: DataFrame
        :return: a DataFrame with computed RPKM
        :rtype: DataFrame
        """

        arguments = locals()
        
        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)
        
        genes_list = DataRownames().apply(data)

        genes_lenght = GenesLenght().apply(self.get_genemodel())
        
        scales = data.sum(axis=0) # Sums the columns

        # Calculating Reads per million (RPM)

        """
        DIVIDE LAS COLUMNAS DE LA MATRIZ ENTRE LOS ESCALARES CALCULADOS
        Es decir, el elemento[0] de sums divide a cada elemento de la columna 0, el [1] a cada elemento de la columna 1
        """

        mat_RPM = data / scales

        # Normalizing to gene length to get RPKM/FPKM

        rpkm = 1000 * (mat_RPM.loc[genes_list] / genes_lenght.loc[genes_list].values)

        return rpkm