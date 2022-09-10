# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

import src.utils as ut
from src.processing import Task
import src.processing as pro
from src.objects import Workflow


class DataObject():
    """This is a conceptual class representation of a data structure
        to save information used on an experiment. With this, the use
        of data or its modification will be easier to do

    :param counts_dframe: That will be the count matrix of the DataObject
    :type counts_dframe: DataFrame 
    :param obs_: That will be the clinical information associated to count matrix
        on DataObject
    :type obs_: DataFrame
    :param vars_: That will be the gene information associated to count matrix
        on DataObject
    :type vars_: DataFrame
    :param log: That will contains a flow that save step by step the changes of
        the DataObject. Useful to replicate an experiment with another information
    :type log: Workflow
    :param unsdict: It is a variable used to save extra data with an unclassified distribution
        on DataObject.
    :type unsdict: dict
    """

    def __init__(self, counts_dframe: ut.pd.DataFrame, obs_= None, vars_ = None, log=Workflow(), unsdict={}):
        """Constructor method
        """

        print("\nCreating DataObject...\n")

        self.datas = {
            "counts": counts_dframe.astype(ut.np.float),
            "dims": counts_dframe.shape,
            "vars": ut.pd.DataFrame(),  # filtering by rows with obs
            "obs": ut.pd.DataFrame(),  # but empty at the beginning to create the key
            "logs": log,  # list of name of preproc
            "uns": unsdict
        }

        # If there is information for observations
        # Number of observations MUST coincide in counts and obs dataframes.
        # Information that it is only in one of the dataframes will be removed.
        # Counts CAN have a bigger dimension than obs taking into account duplicated data!!
        if isinstance(obs_, ut.pd.DataFrame):

            if not obs_.empty:

                rows_o = pro.DataRownames().apply(obs_)

                # Extract a subdf of counts that have the columns of obs
                self.datas["counts"] = pro.DataProjectionSubstring(
                    rows_o, True).apply(self.datas["counts"])

                # Update its dimension
                self.datas["dims"] = self.datas["counts"].shape

                list_columns_counts = pro.DataColumnames().apply(
                    self.datas["counts"])

                self.datas["obs"] = pro.DataSelectionList(
                    list_columns_counts).apply(obs_)

                self.datas["obs"] = pro.DropDuplicates(
                    0, "first", True).apply(self.datas["obs"])

        """
        TO DO: Obs is a list of different dataframes of metadata (e.g: obs for patient, drug, treatment...)
        In this case, there cannot be duplicated names of columns (e.g: two "gender" columns for the same observations)
        
        elif isinstance(obs_, list):
            for o in obs_:
                if isinstance(obs_, ut.pd.DataFrame):
                    print(o.shape)
        """

        # If there is information for variables
        # Number of variables MUST coincide in counts and vars dataframes.
        # Information that it is only in one of the dataframes will be removed.
        # Counts CAN have a bigger dimension than vars taking into account duplicated data!!
        if isinstance(vars_, ut.pd.DataFrame):

            if not vars_.empty:
                rows_v = list(vars_.index.values)

                # Extract a subdf of counts that have the rows of vars
                select_object = pro.DataSelectionList(rows_v)

                self.datas["counts"] = select_object.apply(
                    self.datas["counts"])

                # Update its dimension
                self.datas["dims"] = self.datas["counts"].shape

                rows_v = pro.DataRownames().apply(self.datas["counts"])

                self.datas["vars"] = pro.DataSelectionList(rows_v).apply(vars_)

        """
        TO DO: Obs is a list of different dataframes of metadata
        In this case, there cannot be duplicated names of columns (e.g: two "gene_names" columns for the same variables)
        
        elif isinstance(vars_, list):            
            for v in vars_:
                if isinstance(vars_, ut.pd.DataFrame):
                    print(v.shape)
        """


###############################################################################

#                        GET AND SET FUNCTIONS

###############################################################################

    """
    GET
    """

    def get_datas(self):
        """Returns a dictionary with all information about DataObject

        :return: a dictionary with information read
        :rtype: dict
        """
        return self.datas

    def get_counts(self):
        """Returns the count matrix of the DataObject

        :return: Count Matrix of the DataObject
        :rtype: DataFrame
        """
        return self.datas["counts"]

    def get_counts_dims(self):
        """Returns the dimensions of Count Matrix

        :return: a tuple with dimensions of Count Matrix
        :rtype: tuple
        """
        return self.datas["dims"]

    def get_obs(self):
        """Returns the clinical information of DataObject

        :return: a DataFrame with clinical information
        :rtype: DataFrame
        """
        return self.datas["obs"]

    def get_obs_dims(self):
        """Returns the dimensions of clinical information

        :return: a tuple with dimensions of clinical information
        :rtype: tuple
        """
        return self.datas["obs"].shape

    def get_vars(self):
        """Returns the gene information of DataObject

        :return: a DataFrame with gene information
        :rtype: DataFrame
        """
        return self.datas["vars"]

    def get_vars_dims(self):
        """Returns the dimensions of gene information

        :return: a tuple with dimensions of gene information
        :rtype: tuple
        """
        return self.datas["vars"].shape

    def get_workflow(self):
        """Returns the flow of the DataObject

        :return: a Workflow object type that contains information 
            about the changes of DataObject
        :rtype: Workflow
        """
        return self.datas["logs"]

    def len_log(self):
        """Returns how many step have done

        :return: number of the step done
        :rtype: int
        """
        return len(self.datas["logs"])

    def get_uns(self):
        """Returns an unstructured data of the DataObject

        :return: a dictionary with unclassified information
        :rtype: dict
        """
        return self.datas["uns"]

    """
    SET
    """

    # If the dimensions of counts change, it is needed to modify the affected metadata
    # to make both datasets have the SAME variables / observations.
    # E.g: If a observation is removed from counts, the same observation must be removed
    # for the observation's metadata.
    def set_counts(self, counts: ut.pd.DataFrame):
        """Set DataObject's Count Matrix to new one.
            Control that the rest of the data match with new Count Matrix

        :param counts: new Count Matrix
        :type counts: DataFrame
        """
        self.datas["counts"] = counts

        # To avoid errors, dim of Counts dataframe is updated
        # only if count_data is updated
        self.datas["dims"] = counts.shape

        if isinstance(self.get_obs(), ut.pd.DataFrame):

            if not self.get_obs().empty:

                list_columns_obs = self.counts_columnames()

                self.datas["obs"] = pro.DataSelectionList(
                    list_columns_obs).apply(self.datas["obs"])

                # Remove duplicates because observations cannot have duplicated data

                self.datas["obs"] = pro.DropDuplicates(
                    0, "first", True).apply(self.datas["obs"])

                # Update counts
                rows_o = self.obs_rownames()

                # Make sure counts doesn't have information that there isn't in observations

                self.datas["counts"] = pro.DataProjectionSubstring(
                    rows_o, True).apply(self.datas["counts"])

                self.datas["dims"] = self.datas["counts"].shape

        if isinstance(self.get_vars(), ut.pd.DataFrame):

            if not self.get_vars().empty:

                rows_v = self.counts_rownames()

                self.datas["vars"] = pro.DataSelectionList(
                    rows_v).apply(self.datas["vars"])

                # Remove duplicates

                self.datas["vars"] = pro.DropDuplicates(
                    0, "first").apply(self.datas["vars"])

                # Update counts
                rows_v = self.var_rownames()

                # Make sure counts doesn't have information that there isn't in variables
                self.datas["counts"] = pro.DataSelectionList(
                    rows_v).apply(self.datas["counts"])

                self.datas["dims"] = self.datas["counts"].shape

    def update_counts(self, counts: ut.pd.DataFrame):
        """Set DataObject's Count Matrix to new one.
            It only modifies values inside matrix, neither columns or rows

        :param counts: new values of Count Matrix
        :type counts: DataFrame
        """
        self.datas["counts"] = counts

    # When obs it is updated, it is needed to update counts information to match between observations

    def set_obs(self, obs_):
        """Set DataObject's clinical information to new one.
            Control that the rest of the data match with new clinical information

        :param obs_: new clinical information
        :type obs_: DataFrame
        """
        if isinstance(obs_, ut.pd.DataFrame):

            rows_o = list(obs_.index.values)

            self.datas["counts"] = pro.DataProjectionSubstring(
                rows_o, True).apply(self.datas["counts"])

            self.datas["dims"] = self.datas["counts"].shape

            list_columns_obs = self.counts_columnames()

            # updated obs
            self.datas["obs"] = pro.DataSelectionList(
                list_columns_obs).apply(obs_)

            # Remove duplicates
            self.datas["obs"] = pro.DropDuplicates(
                0, "first").apply(self.datas["obs"])

        elif isinstance(obs_, list):
            for o in obs_:
                if isinstance(obs_, ut.pd.DataFrame):
                    print(o.shape)

        else:
            print("The Obs passed is not correct\n")
            ut.sys.exit(0)

    # When obs it is updated, it is needed to update counts information to match between observations

    def set_var(self, vars_):
        """Set DataObject's gene information to new one.
            Control that the rest of the data match with new gene information

        :param vars_: new gene information
        :type vars_: DataFrame
        """
        if isinstance(vars_, ut.pd.DataFrame):

            rows_v = list(vars_.index.values)

            self.datas["counts"] = pro.DataSelectionList(
                rows_v).apply(self.datas["counts"])

            self.datas["dims"] = self.datas["counts"].shape

            rows_v = self.counts_rownames()

            self.datas["vars"] = pro.DataSelectionList(rows_v).apply(vars_)

            # Remove duplicates

            self.datas["vars"] = pro.DropDuplicates(
                0, "first").apply(self.datas["vars"])

        elif isinstance(vars_, list):
            for v in vars_:
                if isinstance(vars_, ut.pd.DataFrame):
                    print(v.shape)

        else:
            print("The Vars passed is not correct\n")
            ut.sys.exit(0)

    def set_log(self, loglist):
        """Set DataObject's Log to new one.

        :param loglist: new workflow
        :type loglist: Workflow
        """
        self.datas["logs"] = loglist

    def set_uns_data(self, unsdict):
        """Set DataObject's unstructured data to new one.

        :param unsdict: new unstructured data
        :type unsdict: dict
        """
        self.datas["uns"] = unsdict


###############################################################################

#                             BASIC GET

###############################################################################

    """
    COUNTS
    """

    def counts_index_name(self):
        """Returns the index name of Count Matrix

        :return: the name of the index of Count Matrix
        :rtype: str
        """
        return self.datas["counts"].index.name

    def counts_columnames(self):
        """Returns the column names of Count Matrix

        :return: a list with column names of Count Matrix
        :rtype: list
        """
        return list(self.datas["counts"].columns.values)

    def counts_num_columns(self):
        """Returns how many columns have Count Matrix

        :return: number of columns of Count Matrix
        :rtype: int
        """
        return self.datas["dims"][1]

    def counts_rownames(self):
        """Returns the row names of Count Matrix

        :return: a list with row names of Count Matrix
        :rtype: list
        """
        return list(self.datas["counts"].index.values)

    def counts_num_rows(self):
        """Returns how many rows have Count Matrix

        :return: number of rows of Count Matrix
        :rtype: int
        """
        return self.datas["dims"][0]

    """
    OBS
    """

    def obs_index_name(self):
        """Returns the index name of clinical information

        :return: the name of the index of clinical information
        :rtype: str
        """
        return self.datas["obs"].index.name

    def obs_columnames(self):
        """Returns the column names of clinical information

        :return: a list with column names of clinical information
        :rtype: list
        """
        return list(self.datas["obs"].columns.values)

    def obs_num_columns(self):
        """Returns how many columns have clinical information

        :return: number of columns of clinical information
        :rtype: int
        """
        return self.datas["obs"].shape[1]

    def obs_rownames(self):
        """Returns the row names of clinical information

        :return: a list with row names of clinical information
        :rtype: list
        """
        return list(self.datas["obs"].index.values)

    def obs_num_rows(self):
        """Returns how many rows have clinical information

        :return: number of rows of clinical information
        :rtype: int
        """
        return self.datas["obs"].shape[0]

    """
    VAR
    """

    def var_index_name(self):
        """Returns the index name of gene information

        :return: the name of the index of gene information
        :rtype: str
        """
        return self.datas["vars"].index.name

    def var_columnames(self):
        """Returns the column names of gene information

        :return: a list with column names of gene information
        :rtype: list
        """
        return list(self.datas["vars"].columns.values)

    def var_num_columns(self):
        """Returns how many columns have gene information

        :return: number of columns of gene information
        :rtype: int
        """
        return self.datas["vars"].shape[1]

    def var_rownames(self):
        """Returns the row names of gene information

        :return: a list with row names of gene information
        :rtype: list
        """
        return list(self.datas["vars"].index.values)

    def var_num_rows(self):
        """Returns how many row have gene information

        :return: number of row of gene information
        :rtype: int
        """
        return self.datas["vars"].shape[0]


###############################################################################

#                             STR - SUMMARY

###############################################################################

    def __str__(self):
        """str method
        """
        summ = ("\n##############################################")
        summ = summ + ("\nData Object")
        summ = summ + ("\tDimensions: {}".format(self.get_counts_dims()))

        summ = summ + ("\tRow Names ({}): {} ... {}".format(self.counts_num_rows(),
                       self.counts_rownames()[0], self.counts_rownames()[-1]))
        summ = summ + ("\tColumn Names ({}): {} ... {}".format(self.counts_num_columns(),
                       self.counts_columnames()[0], self.counts_columnames()[-1]))

        summ = summ + ("\tVar")

        if self.datas["vars"].empty:
            summ = summ + ("\t\tThere is no data for Vars")
        else:
            summ = summ + ("\t\tDimensions: {}".format(self.get_vars_dims()))

            summ = summ + ("\t\tRow Names ({}): {} ... {}".format(self.var_num_rows(),
                           self.var_rownames()[0], self.var_rownames()[-1]))
            summ = summ + ("\t\tColumn Names ({}): {} ... {}".format(
                self.var_num_columns(), self.var_columnames()[0], self.var_columnames()[-1]))
            # RowData Names is ColumnsName of Vars

        summ = summ + ("\n\tObs")

        if self.datas["obs"].empty:
            summ = summ + ("\t\tThere is no data for observations")
        else:
            summ = summ + ("\t\tDimensions: {}".format(self.get_vars_dims()))

            summ = summ + ("\t\tRow Names ({}): {} ... {}".format(self.obs_num_rows(),
                           self.obs_rownames()[0], self.obs_rownames()[-1]))
            summ = summ + ("\t\tColumn Names ({}): {} ... {}".format(
                self.obs_num_columns(), self.obs_columnames()[0], self.obs_columnames()[-1]))
            # ColData Names is ColumnsName of Obs
        summ = summ + ("\n##############################################")

        return summ

    def summary_object(self):
        """Represent a summary of the DataObject on output
        """

        print("\n##############################################")
        print("\nData Object")
        print("\tDimensions: ", self.get_counts_dims())

        print("\tRow Names ({}): {} ... {}".format(self.counts_num_rows(),
              self.counts_rownames()[0], self.counts_rownames()[-1]))
        print("\tColumn Names ({}): {} ... {}".format(self.counts_num_columns(
        ), self.counts_columnames()[0], self.counts_columnames()[-1]))

        print("\n\tVar")
        if self.datas["vars"].empty:
            print("\t\tThere is no data for Vars")
        else:
            print("\t\tDimensions: ", self.get_vars_dims())

            print("\t\tRow Names ({}): {} ... {}".format(
                self.var_num_rows(), self.var_rownames()[0], self.var_rownames()[-1]))
            print("\t\tColumn Names ({}): {} ... {}".format(
                self.var_num_columns(), self.var_columnames()[0], self.var_columnames()[-1]))
            # RowData Names is ColumnsName of Vars

        print("\n\tObs")
        if self.datas["obs"].empty:
            print("\t\tThere is no data for observations")
        else:
            print("\t\tDimensions: ", self.get_obs_dims())

            print("\t\tRow Names ({}): {} ... {}".format(
                self.obs_num_rows(), self.obs_rownames()[0], self.obs_rownames()[-1]))
            print("\t\tColumn Names ({}): {} ... {}".format(
                self.obs_num_columns(), self.obs_columnames()[0], self.obs_columnames()[-1]))
            # ColData Names is ColumnsName of Obs
        print("\n##############################################")


###############################################################################

#                             EQ - EQUALITY

###############################################################################

    def __eq__(self, other): 
        if not isinstance(other, DataObject):
            return False

        else:
            
            if self.datas["counts"].equals(other.get_counts()):

                if self.datas["dims"] == other.get_counts_dims():
                    
                    if self.datas["vars"].equals(other.get_vars()):
                        
                        if self.datas["obs"].equals(other.get_obs()):
                            
                            if self.datas["uns"] == other.get_uns():
                                
                                return True

            return False



###############################################################################

#                     FUNCTIONS TO WORK WITH THE DATA

###############################################################################


class DataObjectTask(Task):
    """This is a conceptual class representation that organize
        all method associate with DataObject as Task
    """

    def __init__(self):
        """Constructor method
        """
        super().__init__()

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = {}
        return parameters


"""

COUNTS

"""


class DataObjectCounts(DataObjectTask):
    """This is a conceptual class representation that organize
        all method associate with Count Matrix of DataObject as Task
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
#                        BASIC FUNCTIONS FOR COUNTS
# =============================================================================


class SetCounts(DataObjectCounts):
    """Set DataObject's Count Matrix to new one.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param new_counts: new Count Matrix
    :type new_counts: DataFrame
    """

    def __init__(self, new_counts=ut.pd.DataFrame()):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.new_counts = new_counts
        self.index_name = self.new_counts.index.name

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['new_counts'] = self.new_counts
        parameters['index_name'] = self.index_name
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.new_counts = ut.pd.DataFrame(parameters['new_counts'])
        self.new_counts = self.new_counts.reset_index()

        nm = list(self.new_counts.columns.values)
        self.new_counts = self.new_counts.rename(
            columns={nm[0]: parameters['index_name']})
        self.new_counts = self.new_counts.set_index(parameters['index_name'])
        self.index_name = parameters['index_name']

    # Getters

    def get_new_counts(self):
        """Returns the new count matrix

        :return: a dataframe with the new count matrix
        :rtype: DataFrame
        """
        return self.new_counts

    def new_counts_to_json(self):
        """Transform a DataFrame to JSON format.
        """
        # self.new_counts = self.new_counts.to_json()
        self.new_counts = self.new_counts.to_dict()

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "new_counts": {
                    "type": ut.pd.DataFrame,
                    "description": "new content for count matrix of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to change its count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "the same DataObject with count matrix changed"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        aux_self = ut.copy_object(self)

        aux_self.new_counts_to_json()

        dobj.get_workflow().add_function(aux_self, name="Counts - Set",
                                         descp="Set method of the Counts DataFrame.")

        dobj.set_counts(self.get_new_counts())

        return dobj


class UpdateCounts(DataObjectCounts):
    """Set DataObject's Count Matrix to new one but only modify its values.
        In this case, the function not control that the rest of the data match with new Count Matrix
        because any row or column will be modify. Will have the same shape
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param new_counts: new Count Matrix
    :type new_counts: DataFrame
    """

    def __init__(self, new_counts=ut.pd.DataFrame()):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.new_counts = new_counts

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['new_counts'] = self.new_counts
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.new_counts = parameters['new_counts']

    # Getters

    def get_new_counts(self):
        """Returns the new count matrix

        :return: a dataframe with the new count matrix
        :rtype: DataFrame
        """
        return self.new_counts

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "new_counts": {
                    "type": ut.pd.DataFrame,
                    "description": "new values for count matrix of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to change its values of matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "the same DataObject with count matrix changed without modify Vars and Obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        if isinstance(self.get_new_counts(), ut.pd.DataFrame):

            dobj.get_workflow().add_function(self, name="Counts - Update Values",
                                             descp="Update the values of the Counts DataFrame.")

            dobj.update_counts(self.get_new_counts())

        else:
            print("Data must be a DataFrame")
            ut.sys.exit(0)

        return dobj


# =============================================================================
#                       ACCESSING COUNTS DATAFRAME
# =============================================================================


class CountsSelection(DataObjectCounts):
    """Set DataObject's Count Matrix selecting a specific row list.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

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
            "Input Parameters": {
                "row_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of rows name to select the count matrix of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to select specific rows of count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject selected by list on count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Selection",
                                         descp="Extract a sub-DataFrame using a list of rows over the Counts DataFrame.")

        extractor = pro.DataSelectionList(
            self.get_row_list()).apply(dobj.get_counts())

        dobj.set_counts(extractor)

        return dobj


class CountsProjection(DataObjectCounts):  # Columns
    """Set DataObject's Count Matrix selecting a specific column list.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param col_list: A list with the names of the columns to keep
    :type col_list: list
    """

    def __init__(self, col_list=[]):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_list = col_list

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['col_list'] = self.col_list
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_list = parameters['col_list']

    # Getters

    def get_col_list(self):
        """Returns the list with the column names to keep

        :return: a list with column names
        :rtype: list
        """
        return self.col_list

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "col_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of columns name to project the DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to project specific columns of count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject projected by list on count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Projection",
                                         descp="Extract a sub-DataFrame using a list of columns over the Counts DataFrame.")

        extractor = pro.DataProjectionList(
            self.get_col_list()).apply(dobj.get_counts())

        dobj.set_counts(extractor)

        return dobj


class CountsProjectionSubstring(DataObjectCounts):
    """Set DataObject's Count Matrix selecting a specific column list.
        In that case, is not neccesary that name of the column were exactly
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param col_list: A list with the names of the columns to keep
    :type col_list: list
    :param rename: True if the column names of Count Matrix need to be renamed, False otherwise
    :type rename: bool
    """

    def __init__(self, col_list=[], rename=False):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_list = col_list
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
        parameters['col_list'] = self.col_list
        parameters['rename'] = self.rename
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_list = parameters['col_list']
        self.rename = parameters['rename']

    # Getters

    def get_col_list(self):
        """Returns the list with the column names to keep

        :return: a list with column names
        :rtype: list
        """
        return self.col_list

    def get_rename(self):
        """Returns option selected to rename or not column name of Count Matrix

        :return: True if need to rename, False otherwise
        :rtype: bool
        """
        return self.get_rename

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "col_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of columns name to project the DataObject. No need to be the exact name"
                },
                "rename": {
                    "type": bool,
                    "description": "True if the column name of count matrix will be renamed to partial name of col_list, False otherwise"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to project specific columns of count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject projected by list on count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Projection + Substring",
                                         descp="Extract a sub-DataFrame using a list of column name's substrings over the Counts DataFrame.")

        extractor = pro.DataProjectionSubstring(
            self.get_col_list(), self.get_rename()).apply(dobj.get_counts())

        dobj.set_counts(extractor)

        return dobj


# replace nan by mean
class CountsReplaceNan(DataObjectCounts):
    """Replace nan values to mean of the row of Count Matrix
        Only modifies values of Count Matrix.
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
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
            "Input Parameters": {
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to replace nan values to mean on count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with nan of count matrix replaced by mean"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Replace Nans by Mean",
                                         descp="Replace all NaN values from the Counts DataFrame.")

        data_tmp = pro.FillNan().apply(dobj.get_counts())

        dobj.update_counts(data_tmp)

        return dobj


class CountsDropDuplicates(DataObjectCounts):
    """Drop DataObject's Count Matrix duplicates.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Count Matrix.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool
    """

    def __init__(self, axis=0, keep=False, by_name=False):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis  # axis 0 rows, 1 columns
        self.keep = keep  # False delete both duplicates
        self.by_name = by_name  # if True, only take into account duplicates by column/row names

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
        """Returns which duplicates will keep on Count Matrix

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
            "Input Parameters": {
                "axis": {
                    "type": int,
                    "description": "indicated 0 to row or 1 to columns"
                },
                "keep": {
                    "type": (str, bool),
                    "description": "\"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows"
                },
                "by_name": {
                    "type": bool,
                    "description": "True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to drop duplicates on count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject without duplicates rows or columns on count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Drop Duplicates",
                                         descp="Remove the duplicates from the Counts DataFrame.")

        drop = pro.DropDuplicates(self.get_axis(), self.get_keep(
        ), self.get_by_name()).apply(dobj.get_counts())

        dobj.set_counts(drop)

        return dobj


class CountsDropValues(DataObjectCounts):
    """Drop DataObject's Count Matrix values.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param to_delete: value to be delete.
    :type to_delete: str, float, int, bool, list, ndarray, 
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param method: all to check all values are the same, any if a partial value match
    :type method: str
    """

    def __init__(self, to_delete = ut.np.nan, axis=0, method="all"):

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
            "Input Parameters": {
                "to_delete": {
                    "type": (str, float, int, bool, list, ut.np.ndarray),
                    "description": "value to be delete"
                },
                "axis": {
                    "type": int,
                    "description": "Indicates 0 to row and 1 to columns"
                },
                "method": {
                    "type": str,
                    "description": "all to check all values are the same, any if a partial value match"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to drop values on count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject without that values on rows or columns on count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Drop Values",
                                         descp="Remove the value(s) in the rows/columns from the Counts DataFrame.")

        dropn = pro.DropValues(self.get_to_delete(), self.get_axis(
        ), self.get_method()).apply(dobj.get_counts())

        dobj.set_counts(dropn)

        return dobj


class CountsDropNanByThresh(DataObjectCounts):
    """Drop DataObject's Count Matrix values limited by Threshold.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param thresh: threshold to evaluate element. If value less than threshold, that will be selected to drop
    :type thresh: float, int
    """

    def __init__(self, axis=0, thresh=0):

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
            "Input Parameters": {
                "axis": {
                    "type": int,
                    "description": "indicates 0 to rows or 1 to columns"
                },
                "thresh": {
                    "type": (float, int),
                    "description": "threshold to evaluate element. If value less than threshold, that will be selected to drop"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to drop values"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with drop element checking a threshold"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Drop Nan values",
                                         descp="Replace value(s) of the Counts DataFrame.")

        dropthresh = pro.DropNanByThresh(
            self.get_axis(), self.get_thresh()).apply(dobj.get_counts())

        dobj.set_counts(dropthresh)

        return dobj


class CountsReplace(DataObjectCounts):
    """Replace DataObject's Count Matrix values with another.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool
    """

    def __init__(self, to_replace="Unknown", replaced_by=ut.np.nan):

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
            "Input Parameters": {
                "to_replace": {
                    "type": (str, float, int, bool, list, ut.np.ndarray),
                    "description": "values to be replaced"
                },
                "replaced_by": {
                    "type": (str, float, int, bool),
                    "description": "new value to replaced the value before"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to replace values"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with replaced values"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Replace Values",
                                         descp="Replace value(s) of the Counts DataFrame.")

        replace = pro.Replace(self.get_to_replace(),
                              self.get_replaced_by()).apply(dobj.get_counts())

        dobj.set_counts(replace)

        return dobj


class CountsTranspose(DataObjectCounts):

    def __init__(self, index_name=""):
        """Do transpose to DataObject's Count Matrix
            Control that the rest of the data match with new Count Matrix
            This class is a Task which allows generate a Workflow
            When this class is called, save his instance on DataObject's Log

        :param index_name: the name of the index before transpose to keep it
        :type index_name: str, int
        """
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

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "index_name": {
                    "type": (str, int),
                    "description": "name of the index if exists"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to do a transpose on count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with transpose count matrix"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - Transpose",
                                         descp="Calculates the transpose of the Counts DataFrame.")

        transpose = pro.Transpose(
            self.get_index_name()).apply(dobj.get_counts())

        dobj.set_counts(transpose)

        return dobj


class CountsCPM(DataObjectCounts):
    """Normalize DataObject's Count Matrix using CPM method
        It is not neccesary to control that the rest of the data match with new Count Matrix.
        It only modify values inside Count Matrix.
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param log_method: True if CPM will do with logaritmic alghoritm, False otherwise
    :type log_method: bool
    :param prior_count: If log is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size
    :type prior_count: int
    """

    def __init__(self, log_method=False, prior_count=2):

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

    # Getters

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
            "Input Parameters": {
                "log_method": {
                    "type": bool,
                    "description": "True if CPM will do with logaritmic alghoritm, False otherwise"
                },
                "prior_count": {
                    "type": int,
                    "description": "If log is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to normalize its count matrix"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with normalized count matrix by one of two options"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Counts - CPM",
                                         descp="Operates a Count-per-Million (CPM) normalization over the Counts DataFrame.")

        norm_cpm = pro.CPM(self.get_log_method(),
                           self.get_prior_count()).apply(dobj.get_counts())

        dobj.update_counts(norm_cpm)

        return dobj


"""

OBSERVATIONS

"""


class DataObjectObs(DataObjectTask):
    """This is a conceptual class representation that organize
        all method associate with Observation (Clinical Information) of DataObject as Task
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


class SetObs(DataObjectObs):
    """Set DataObject's Clinical Information to new one.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param new_obs: new Clinical Information
    :type new_obs: DataFrame
    """

    def __init__(self, new_obs=ut.pd.DataFrame()):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.new_obs = new_obs

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['new_obs'] = self.new_obs
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.new_obs = parameters['new_obs']

    # Getters

    def get_new_obs(self):
        """Returns the new observations

        :return: a dataframe with the new clinical information
        :rtype: DataFrame
        """
        return self.new_obs

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "new_obs": {
                    "type": ut.pd.DataFrame,
                    "description": "new content for obs of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to change its obs"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "the same DataObject with obs changed"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Set",
                                         descp="Set method of the Obs DataFrame.")

        dobj.set_obs(self.get_new_obs())

        return dobj


# =============================================================================
#                           ACCESSING OBS DATAFRAME
# =============================================================================


class ObsSelection(DataObjectObs):
    """Set DataObject's Observations selecting a specific row list.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

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
            "Input Parameters": {
                "row_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of rows name to select the obs of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to select specific rows of obs"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject selected by list on obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Selection",
                                         descp="Extract a sub-DataFrame using a list of rows over the Obs DataFrame.")

        extractor = pro.DataSelectionList(
            self.get_row_list()).apply(dobj.get_obs())

        dobj.set_obs(extractor)

        return dobj


class ObsProjection(DataObjectObs):  # Columns
    """Set DataObject's Observations selecting a specific column list.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param col_list: A list with the names of the columns to keep
    :type col_list: list
    """

    def __init__(self, col_list=[]):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_list = col_list

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['col_list'] = self.col_list
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_list = parameters['col_list']

    # Getters

    def get_col_list(self):
        """Returns the list with the column names to keep

        :return: a list with column names
        :rtype: list
        """
        return self.col_list

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "col_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of columns name to project the obs of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to project specific rows of obs"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject projected by list on obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Projection",
                                         descp="Extract a sub-DataFrame using a list of columns over the Obs DataFrame.")

        extractor = pro.DataProjectionList(
            self.get_col_list()).apply(dobj.get_obs())

        dobj.set_obs(extractor)

        return dobj


class ObsAddColumn(DataObjectObs):
    """Add to DataObject's Observations a new column
        It is not neccesary control that the rest of the data match with new Observations.
        Add new column doesn't affect the rest of the DataObject
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of Observations
    :type values: list, ndarray
    """

    def __init__(self, name_column="", values=[]):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.name_column = name_column
        self.keep = values

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
            "Input Parameters": {
                "name_column": {
                    "type": str,
                    "description": "name of the new column"
                },
                "values": {
                    "type": (list, ut.np.ndarray),
                    "description": "values of the new column. Must have the same lenght as number of rows of obs"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "new column with its values will be added to this dataframe"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with the new column added to obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        added = pro.AddColumn(self.get_name_column(),
                              self.get_values()).apply(dobj.get_obs())

        # Add a column doesn't affect the rest of the DataObject
        dobj.set_obs(added)

        return dobj


class ObsReplace(DataObjectObs):
    """Replace DataObject's Observations values with another.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool
    """

    def __init__(self, to_replace="Unknown", replaced_by=ut.np.nan):

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
            "Input Parameters": {
                "to_replace": {
                    "type": (str, float, int, bool, list, ut.np.ndarray),
                    "description": "values to be replaced"
                },
                "replaced_by": {
                    "type": (str, float, int, bool),
                    "description": "new value to replaced the value before"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to replace values on Obs"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with replaced values on Obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """
        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Replace Values",
                                         descp="Replace value(s) of the Obs DataFrame.")

        replace = pro.Replace(self.get_to_replace(),
                              self.get_replaced_by()).apply(dobj.get_obs())

        dobj.set_obs(replace)

        return dobj


class ObsDropDuplicates(DataObjectObs):
    """Drop DataObject's Observations duplicates.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Observations.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool
    """

    def __init__(self, axis=0, keep=False, by_name=False):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis  # axis 0 rows, 1 columns
        self.keep = keep  # False delete both duplicates
        self.by_name = by_name  # if True, only take into account duplicates by column/row names

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
        """Returns which duplicates will keep on Observations

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
            "Input Parameters": {
                "axis": {
                    "type": int,
                    "description": "indicated 0 to row or 1 to columns"
                },
                "keep": {
                    "type": (str, bool),
                    "description": "\"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows"
                },
                "by_name": {
                    "type": bool,
                    "description": "True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to drop duplicates on Obs"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject without duplicates rows or columns on Obs"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Drop Duplicates",
                                         descp="Remove the duplicates from the Obs DataFrame.")

        drop = pro.DropDuplicates(
            self.get_axis(), self.get_keep(), self.get_by_name()).apply(dobj.get_obs())

        dobj.set_obs(drop)

        return dobj


"""

VARIABLES

"""


class DataObjectVars(DataObjectTask):
    """This is a conceptual class representation that organize
        all method associate with Variables of DataObject as Task
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


class SetVars(DataObjectVars):
    """Set DataObject's Variables to new one.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param new_var: new Gene Information
    :type new_var: DataFrame
    """

    def __init__(self, new_var=ut.pd.DataFrame()):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.new_var = new_var

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['new_var'] = self.new_var
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.new_var = parameters['new_var']

    # Getters

    def get_new_var(self):
        """Returns the new variables

        :return: a dataframe with the new gene information
        :rtype: DataFrame
        """
        return self.new_var

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "new_var": {
                    "type": ut.pd.DataFrame,
                    "description": "new content for Vars of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to change its Vars"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "the same DataObject with vars changed"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject

        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Vars - Set",
                                         descp="Set method of the Vars DataFrame.")

        dobj.set_var(self.get_new_var())

        return dobj


# =============================================================================
#                           ACCESSING VAR DATAFRAME
# =============================================================================


class VarsSelection(DataObjectVars):
    """Set DataObject's Variables selecting a specific row list.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

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
            "Input Parameters": {
                "row_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of rows name to select the Vars of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to select specific rows of Vars"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject selected by list on Vars"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Vars - Selection",
                                         descp="Extract a sub-DataFrame using a list of rows over the Vars DataFrame.")

        extractor = pro.DataSelectionList(
            self.get_row_list()).apply(dobj.get_vars())

        dobj.set_var(extractor)

        return dobj


class VarsProjection(DataObjectVars):  # Columns
    """Set DataObject's Variables selecting a specific column list.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param col_list: A list with the names of the columns to keep
    :type col_list: list
    """

    def __init__(self, col_list=[]):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.col_list = col_list

    # Parameters

    def get_parameters(self) -> dict:
        """Returns a dictionary with all parameters needed to
            initialize the class object to apply a Task.
            It allows to implement Reflection

        :return: a dictionary with all parameters
        :rtype: dict
        """
        parameters = super().get_parameters()
        parameters['col_list'] = self.col_list
        return parameters

    def set_parameters(self, parameters: dict):
        """Update the initial parameters to specific arguments
            for a Task necessary for a Task to apply its functionality.

        :param parameters: a dictionary with specific parameters
        :type parameters: dict
        """
        super().set_parameters(parameters)
        self.col_list = parameters['col_list']

    # Getters

    def get_col_list(self):
        """Returns the list with the column names to keep

        :return: a list with column names
        :rtype: list
        """
        return self.col_list

    def get_metadata(self):
        """Returns a dictionary denoting the type of parameters the class will receive.
            This allows check parameters before execution and it allows 
            the user to know the information about them.

        :return: a dictionary with Input Parameters information, Apply Input parameters information
            and the return information
        :rtype: dict
        """
        return {
            "Input Parameters": {
                "col_list": {
                    "type": (list, ut.np.ndarray),
                    "description": "list of columns name to project the Vars of DataObject"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to project specific rows of Vars"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject projected by list on Vars"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Vars - Selection",
                                         descp="Extract a sub-DataFrame using a list of rows over the Vars DataFrame.")

        extractor = pro.DataProjectionList(
            self.get_col_list()).apply(dobj.get_vars())

        dobj.set_var(extractor)

        return dobj


class VarsAddColumn(DataObjectVars):
    """Add to DataObject's Variables a new column
        It is not neccesary control that the rest of the data match with new Variables.
        Add new column doesn't affect the rest of the DataObject
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of Variables
    :type values: list, ndarray
    """

    def __init__(self, name_column="", values=[]):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.name_column = name_column
        self.keep = values

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
            "Input Parameters": {
                "name_column": {
                    "type": str,
                    "description": "name of the new column"
                },
                "values": {
                    "type": (list, ut.np.ndarray),
                    "description": "values of the new column. Must have the same lenght as number of rows of Vars"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "new column with its values will be added to this dataframe"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with the new column added to Vars"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Add New Column",
                                         descp="Add a new column in the Vars DataFrame.")

        added = pro.AddColumn(self.get_name_column(),
                              self.get_values()).apply(dobj.get_vars())

        # Add a column doesn't affect the rest of the DataObject
        dobj.set_var(added)

        return dobj


class VarsReplace(DataObjectVars):
    """Replace DataObject's Variables values with another.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool
    """

    def __init__(self, to_replace="Unknown", replaced_by=ut.np.nan):

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
            "Input Parameters": {
                "to_replace": {
                    "type": (str, float, int, bool, list, ut.np.ndarray),
                    "description": "values to be replaced"
                },
                "replaced_by": {
                    "type": (str, float, int, bool),
                    "description": "new value to replaced the value before"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to replace values on Vars"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject with replaced values on Vars"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Obs - Replace Values",
                                         descp="Replace value(s) of the Vars DataFrame.")

        replace = pro.Replace(self.get_to_replace(),
                              self.get_replaced_by()).apply(dobj.get_vars())

        dobj.set_var(replace)

        return dobj


class VarsDropDuplicates(DataObjectVars):
    """Drop DataObject's Variables duplicates.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Observations.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool
    """

    def __init__(self, axis=0, keep=False, by_name=False):

        arguments = locals()

        inputs_params = self.get_metadata()["Input Parameters"]

        ut.check_arguments(arguments, inputs_params)

        super().__init__()
        self.axis = axis  # axis 0 rows, 1 columns
        self.keep = keep  # False delete both duplicates
        self.by_name = by_name  # if True, only take into account duplicates by column/row names

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
        """Returns which duplicates will keep on Variables

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
            "Input Parameters": {
                "axis": {
                    "type": int,
                    "description": "indicated 0 to row or 1 to columns"
                },
                "keep": {
                    "type": (str, bool),
                    "description": "\"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows"
                },
                "by_name": {
                    "type": bool,
                    "description": "True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not"
                }
            },
            "Apply Input": {
                "dobj": {
                    "type": DataObject,
                    "description": "a DataObject to drop duplicates on Vars"
                }
            },
            "Output": {
                "return": {
                    "type": DataObject,
                    "description": "a DataObject without duplicates rows or columns on Vars"
                }
            }
        }

    def show_metadata(self):
        """It only print on output the parameters information
        """
        ut.print_json(self.get_metadata())

    # Execution

    def apply(self, dobj):
        """It is the main function of the class. 
            It reproduces the corresponding task of the class 
            for which it was created and update the Workflow for the DataObject

        :param dobj: The current DataObject to be process
        :type dobj: DataObject
        :return: The DataObject passed with the Task applied
        :rtype: DataObject
        """

        arguments = locals()

        inputs_params = self.get_metadata()["Apply Input"]

        ut.check_arguments(arguments, inputs_params)

        dobj.get_workflow().add_function(self, name="Vars - Drop Duplicates",
                                         descp="Remove the duplicates from the Vars DataFrame.")

        drop = pro.DropDuplicates(
            self.get_axis(), self.get_keep(), self.get_by_name()).apply(dobj.get_vars())

        dobj.set_var(drop)

        return dobj
