# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

import math
from src import utils as ut
from src import processing as pro
from src import dataobject as dobj
from src import visualize as vis
from src.processing import Task
from src.dataobject import DataObject
from src.etl import ProcessGDC
from src.objects import GDCQuery
from src.objects import DataProject
from src.objects import FileProject
from src.objects import Workflow
from src.model import ModelSelection
from src.model import LogisticRegression
from src.model import RandomForest
from src.model import ScikitLearnModel
from src.model import SupportVectorClassif
from src.model import NeuralNetwork



"""
######################################################

                    TASK

######################################################
"""


def from_dict(json_task):
    """Returns a object Task replicate from json or dictionary

    :return: a object Task replicated
    :rtype: Task
    """
    return Task.from_dict(json_task)


def workflow_to_json(workfw : Workflow, path = ""):
    """Generate an external file to save every Log object of Workflow

    :param path: file where dictionary generates will be saved
    :type: str
    """
    workfw.generate_json(path)


def workflow_from_json(path = ""):
    """Open a json file with some Log saved as dictionary
        and creates a new Workflow throught them

    :param path: The file that contains object Task as dictionary for lines
    :type: str

    :return: a replicate Workflow from json dictionary
    :rtype: Workflow    
    """
    
    flow = Workflow()
    opened = "workflow.json"

    if path != "":
        opened = path

    with open(opened, 'r') as handle:
        task_json = ut.json.load(handle)
        
        for tk in task_json:
            flow_task = from_dict(tk)

            flow.add_function(flow_task)

    return flow
    

def replicate_workflow(workflow, data_object):
    """Replicate a workflow to obtain the same result as previous experiment

    :param workflow: a flow of another object that contains step of its experiment
    :type workflow: Workflow
    :param data_object: a DataObject to which the same process will be applied through the Workflow
    :type data_object: DataObject

    :return: a DataObject
    """
    return workflow.apply(data_object)



"""
######################################################

                    VISUALIZE

######################################################
"""


def show_figure(fig, xlabel = "x axis", ylabel = "y axis", x_ini_range = None, x_fin_range = None,
                    y_ini_range = None, y_fin_range = None, legend = True, legend_title = "", 
                    vheight = 600, vwidth = 800, title = "Plot of the figure"):
    """Show in the screen the figure plot with the specified figured parameters.

    :param fig: Figure to update.
    :type fig: Figure
    :param xlabel: Name of the X axis. Defaults to `x axis`.
    :type xlabel: str
    :param ylabel: Name of the Y axis. Defaults to `y axis`.
    :type ylabel: str
    :param x_ini_range: Initial value of X axis range. Defaults to None.
    :type x_ini_range: int
    :param x_fin_range: Final value of X axis range. Defaults to None.
    :type x_fin_range: int
    :param y_ini_range: Initial value of Y axis range. Defaults to None.
    :type y_ini_range: int
    :param y_fin_range: Final value of Y axis range. Defaults to None.
    :type y_fin_range: int
    :param legend: Indicates if the figure will have legend or not. Defaults to `True`.
    :type legend: bool
    :param legend_title: Name of the legend's label
    :type legend_title: str
    :param vheight: Height of the figure as image. Defaults to 600.
    :type vheight: int
    :param vwidth: Width of the figure as image. Defaults to 800.
    :type vwidth: int
    :param title: Title of the figure. Defaults to `Plot of the figure`.
    :type title: str

    :return: The figure with the layout's features updated.
    :rtype: Figure
    """

    return vis.show_figure(fig, xlabel = xlabel, ylabel = ylabel, x_ini_range = x_ini_range, x_fin_range = x_fin_range,
                    y_ini_range = y_ini_range, y_fin_range = y_fin_range, legend = legend, legend_title = legend_title,
                    vheight = vheight, vwidth = vwidth, title = title)


def save_image(fig, fig_name = "fig", img_format = "png", os_path = "images", 
                   xlabel = "x axis", ylabel = "y axis", x_ini_range = None, x_fin_range = None,
                   y_ini_range = None, y_fin_range = None, legend = True, legend_title = "", vheight = 600, vwidth = 800, 
                   title = "Plot of the figure"):
    """Save a figure as image on disk.

    :param fig: Figure to update.
    :type fig: Figure
    :param fig_name: Name of file.
    :type fig_name: str
    :param img_format: Format of the file saved. Available formats: 'png', 'jpeg', 
        'webp', 'svg', 'pdf'.
    :type img_format: str
    :param os_path: Path to save the image.
    :type os_path: str
    :param xlabel: Name of the X axis. Defaults to `x axis`.
    :type xlabel: str
    :param ylabel: Name of the Y axis. Defaults to `y axis`.
    :type ylabel: str
    :param x_ini_range: Initial value of X axis range. Defaults to None.
    :type x_ini_range: int
    :param x_fin_range: Final value of X axis range. Defaults to None.
    :type x_fin_range: int
    :param y_ini_range: Initial value of Y axis range. Defaults to None.
    :type y_ini_range: int
    :param y_fin_range: Final value of Y axis range. Defaults to None.
    :type y_fin_range: int
    :param legend: Indicates if the figure will have legend or not. Defaults to `True`.
    :type legend: bool
    :param legend_title: Name of the legend's label
    :type legend_title: str
    :param vheight: Height of the figure as image. Defaults to 600.
    :type vheight: int
    :param vwidth: Width of the figure as image. Defaults to 800.
    :type vwidth: int
    :param title: Title of the figure. Defaults to `Plot of the figure`.
    :type title: str
    """

    vis.save_image(fig, fig_name = fig_name, img_format = img_format, os_path = os_path, 
                    xlabel = xlabel, ylabel = ylabel, x_ini_range = x_ini_range, x_fin_range = x_fin_range,
                    y_ini_range = y_ini_range, y_fin_range = y_fin_range, legend = legend, legend_title = legend_title,
                    vheight = vheight, vwidth = vwidth, title = title)


def show_image_web(fig):
    """Show an image on a web browser. If there is not an available web browser,
        it shows the image on the default application to visualize images.

    :param fig: Figure to dislay.
    :type fig: Figure
    """

    vis.show_image_web(fig)


def bar_plot(data, x = None, y = None):
    """Create a bar diagram with the data received as the input.

    :param data: Data values to create the diagram.
    :type data: DataFrame
    :param x: Name of X axis. Defaults to None.
    :type x: str
    :param y: Name of Y axis. Defaults to None.
    :type y: str
    
    :return: Figure that contains the bar plot
    :rtype: Figure
    """

    return vis.bar_plot(data, x = x, y = y)


def kde_plot(data):
    """Creates a Kernel Density Estimation (kde) diagram with the data received as the input.
        A kernel density estimate (KDE) plot is a method for visualizing the distribution 
        of observations in a dataset, analagous to a histogram.

    :param data: Data values to create the diagram.
    :type data: DataFrame
    
    :return: Figure that contains the KDE plot.
    :rtype: Figure
    """

    return vis.kde_plot(data)


def box_plot(data, x = None, y = None, median = False):
    """Creates a box plot with the data received as the input.
        If "median" parameter is true, the function will display a line with the median value.
        If x and y are not None, they must be name of row and column of the data.

    :param data: The data used to create the plot.
    :type data: DataFrame
    :param x: Name of X axis. Defaults to None.
    :type x: str
    :param y: Name of Y axis. Defaults to None.
    :type y: str
    :param median: Indicates if the diagram will show a median line. 
        Defaults to `False`.
    :type median: bool

    :return: Figure that contains the box plot.
    :rtype: Figure
    """

    return vis.box_plot(data, x = x, y = y, median = median)



def mds_plot(data, color_list = [], clinical_info = [], symbols = [], text_plot = True):
    """Creates a multidimensional diagram plot with the data received as the input.
    If "color_list" parameter is empty, colors will be formed automatically by function,
    otherwise, lenght of them must be the same as lenght of data columns
    If "text_plot" is true, the name of the columns will appear over the point its point on the plot,
    if False will only appear the point

    :param data: The data used to create the plot.
    :type data: DataFrame
    :param color_list: A list of colors to be used on plot
    :type color_list: list
    :param clinical_info: A classification of the samples to represent them by types
    :type clinical_info: list
    :param symbols: A classification of the samples to represent them by symbols
    :type symbols: list
    :param text_plot: Indicates if name of column should appear over its representation on plot
    :type text_plot: bool

    :return: Figure that contains the mds plot.
    :rtype: Figure
    """

    return vis.mds_plot(data = data, color_list = color_list, clinical_info = clinical_info, symbols = symbols, text_plot = text_plot)



def volcano_plot(data, clinical_data, grouped_by, group_1, group_2):
    """Creates a volcano plot with the data received as the input.
        To do a volcano plot is necessary to indicates two types of information
        collected in clinical data.
        It will be neccesary calculated a stadistic information

    :param data: The data used to create the plot.
    :type data: DataFrame
    :param clinical_data: A dataframe with clinical information of the count matrix
    :type clinical_data: DataFrame
    :param grouped_by: Name of the clinical data column to organize the plot
    :type grouped_by: str
    :param group_1: A type of the clinical data column selected
    :type group_1: str
    :param group_2: A second type of the clinical data column selected
    :type group_2: str

    :return: Figure that contains the volcano plot.
    :rtype: Figure
    """

    return vis.volcano_plot(data, clinical_data, grouped_by, group_1, group_2)


def heatmap(x, y, z, color = 'YlOrRd'):
    """Create a heatmap with the data received as the input.

    :param x: The data to create the heatmap.
    :type x: list
    :param y: The data to create the heatmap.
    :type y: list
    :param z: The data to create the heatmap.
    :type z: list
    :param color: color spectrum used on heatmap
    :type color: str
    
    :return: Figure that contains the heatmap.
    :rtype: Figure
    """

    return vis.heatmap(x, y, z, color = color)


def clustermap(data, color = 'YlOrRd'):
    """Create a clustermap with the data received as the input.
        A clustermap order data by similarity. 
        This reorganizes the data for the rows and columns and displays 
        similar content next to one another for even more 
        depth of understanding the data.

    :param data: The data to create the clustermap.
    :type data: DataFrame
    :param color: color spectrum used on heatmap of clustermap
    :type color: str
    
    :return: Figure that contains the clustermap.
    :rtype: Figure
    """

    return vis.clustermap(data)


def duplicated_corr_plot(data_duplicated, ini_row = 1001, fin_row = 2000):
    """Create a diagram to display the correlation between duplicate variables.
        In case the original dataFrame is set as the input, the method will be check
        for duplicates. If the DataFrame contains only the duplicates, this checking
        will not affect the result.
        The layout wil be created the most squared and symetric as possible.

    :param data_duplicated: The duplicated data used to create the diagram
    :type data_duplicated: DataFrame
    :param ini_row: First row to select the data will be used. Defaults to 1001.
    :type ini_row: int
    :param fin_row: Final row to select the data will be used. Dafults to 2000.
    :type fin_row: int
    
    :return: figure that contains a diagram with duplicated data of a correlation
    :rtype: figure
    """

    return vis.duplicated_corr_plot(data_duplicated, ini_row = ini_row, fin_row = fin_row)



# =============================================================================
#                      MACHINE LEARNING VISUALIZATIONS
# =============================================================================  



def plot_prec_recall_vs_thresh(testy, predictions):
    """Creates the curve between Precision-Real and Thresholds.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list

    :return: Figure that contains the plot showing the precision recall curve over thresholds.
    :rtype: Figure

    """

    return vis.plot_prec_recall_vs_thresh(testy, predictions)


def plot_roc(testy, predictions):
    """Creates a ROC Curve.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list

    :return: Figure that contains the ROC Curve of the Real Value and Predictions.
    :rtype: Figure
    """

    return vis.plot_roc(testy, predictions)


def plot_prc(testy, predictions):
    """Creates a Precision-Recall (PR) Curve.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list
    
    :return: Figure that contains the PR Curve of the Real Value and Predictions.
    :rtype: Figure
    """

    return vis.plot_prc(testy, predictions)


def plot_confusion_matrix(matrix, label_list = None, color = "Viridis"):
    """Create a figure with the Confusion Matriz as Heatmap.

    :param matrix: Contains the elements and data of a confusion matrix.
    :type matrix: list[list]
    :param label_list: Used as value of axis. Defaults to None.
    :type label_list: list
    :param color: color spectrum used on heatmap. Defaults to `Viridis`.
    :type color: list
    
    :return: Figure that contains the heatmap showing the colored confusion matrix.
    :rtype: Figure
    """

    return vis.plot_confusion_matrix(matrix, label_list = label_list, color = color)



"""
######################################################

                    PROCESSING

######################################################
"""

# =============================================================================
#                               ACCESS
# =============================================================================


def columnames(data=None):
    """Return on a list the name of the columns of the DataFrame

    :param data: a dataframe to obtain the name of its columns
    :type data: DataFrame
    
    :return: a list with the column names of the data
    :rtype: list
    """

    return pro.DataColumnames().apply(data)


def rownames(data=None):
    """Return on a list the name of the rows of the DataFrame

    :param data: a dataframe to obtain the name of its rows
    :type data: DataFrame
    
    :return: a list with the row names of the data
    :rtype: list
    """
    return pro.DataRownames().apply(data)


def explain_variable(samples, var_id):
    """Return information about a specific variable of a DataFrame

    :param samples: a dataframe with information of variables
    :type data: DataFrame
    :param var_id: specific identificator of a variable
    :type var_id: str, int
    
    :return: specific information of a variable
    :rtype: series
    """

    return pro.DataExplainVariable(var_id).apply(samples)


def explain_variable_colname(samples, var_id, colnm):
    """Return specific field about a specific variable of a DataFrame

    :param samples: a dataframe with information of variables
    :type data: DataFrame
    :param var_id: specific identificator of a variable
    :type var_id: str, int
    :param colnm: specific column name of the DataFrame (field)
    :type colnm: str, int
    
    :return: specific field of information of a variable
    :rtype: series
    """

    return pro.DataExplainVariableColname(var_id, colnm).apply(samples)


def explain_all_variable_colname(samples, colnm):
    """Return specific field about all variables of a DataFrame

    :param samples: a dataframe with information of variables
    :type data: DataFrame
    :param colnm: specific column name of the DataFrame (field)
    :type colnm: str, int
    
    :return: a dataframe with specific field of information about all variables
    :rtype: DataFrame
    """
    return pro.DataExplainAllVariableColname(colnm).apply(samples)



# =============================================================================
#                               CHECKING
# =============================================================================


def check_element(list_ = [], element = ""):
    """Check if a given element is on a list or array
        
    :param list_: The current list to be process
    :type list_: list
    :param element: element to find on a list
    :type element: int, float, str, bool

    :return: True if element is on a list, False otherwise
    :rtype: bool
    """
    return pro.CheckElement(element).apply(list_)


def check_all_element(list_ = [], sublist = []):
    """Check if a given list of elements are on a list or array
        
    :param list_: The current list to be process
    :type list_: list
    :param sublist: list of element to find
    :type sublist: list, ndarray

    :return: True if all elements is on a list, False otherwise
    :rtype: bool
    """
    return pro.CheckAllElement(sublist).apply(list_)


def check_sub_element(list_ = [], partial_sublist = []):
    """It receives a list with elements where it is only needed to find 
        a SUBSTRING of the name to be valid.
    
    :param list_: The current list to be process
    :type list_: list
    :param partial_sublist: list of element to find without need to match all name
    :type partial_sublist: list, ndarray

    :return: a list of coincidence with the large name
    :rtype: list, ndarray
    """
    return pro.CheckSubElement(partial_sublist).apply(list_)


def check_sub_element_short(list_ = [], partial_sublist = []):
    """It receives a list with elements where it is only needed to find 
        a SUBSTRING of the name to be valid, but returns the short name founded
    
    :param list_: The current list to be process
    :type list_: list
    :param partial_sublist: list of element to find without need to match all name
    :type partial_sublist: list, ndarray

    :return: a list of coincidence with the short name
    :rtype: list, ndarray
    """
    return pro.CheckSubElementShort(partial_sublist).apply(list_)



# =============================================================================
#                               EXTRACTION
# =============================================================================


# Projection means choosing which columns (or expressions) the query shall return.

def data_projection_list(data=None, column_list=[]):
    """Extract a Sub-DataFrame selecting a specific column list.
        
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param column_list: A list with the names of the columns to keep
    :type column_list: list
    :return: a Sub-DataFrame with selected columns
    :rtype: DataFrame
    """
    return pro.DataProjectionList(column_list).apply(data)


def data_projection_index(data=None, indx=0):
    """Extract a Sub-DataFrame selecting a specific column by index.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param indx: a number of a column. Must be on range of lenght columns
    :type indx: int

    :return: a Sub-DataFrame with selected column
    :rtype: DataFrame
    """
    return pro.DataProjectionIndex(indx).apply(data)


def data_projection_name(data=None, name = ""):
    """Extract a Sub-DataFrame selecting a specific column by name.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param name: a name of a column. Must exists
    :type name: str

    :return: a Sub-DataFrame with selected column
    :rtype: DataFrame
    """
    return pro.DataProjectionName(name).apply(data)


def data_projection_range(data=None, ini_column = 0, fini_column = -1):
    """Extract a Sub-DataFrame selecting a range of columns
        
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param ini_column: a initial position to project. Must be on range of lenght columns
    :type ini_column: int
    :param fini_column: a final position to project. Must be on range of lenght columns
    :type fini_column: int

    :return: a Sub-DataFrame with the selected columns in an interval
    :rtype: DataFrame
    """
    return pro.DataProjectionRange(ini_column, fini_column).apply(data)


def data_projection_filter(data=None, filter_ = None):
    """Extract a Sub-DataFrame selecting columns by filter.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param filter_: a filter to do the projection by columns
    :type filter_: pd.Series, pd.DataFrame

    :return: a Sub-DataFrame after apply a filter by columns
    :rtype: DataFrame
    """
    return pro.DataProjectionFilter(filter_).apply(data)


def data_projection_substring(data=None, column_list=[], rename = False):
    """Extract a Sub-DataFrame selecting a specific column list no matter if name doesn't match complete.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param column_list: A list with the names of the columns to keep
    :type column_list: list
    :param rename: check if should rename the column to the short name. True, then rename, False, not
    :type rename: bool

    :return: a Sub-DataFrame with selected columns
    :rtype: DataFrame
    """
    return pro.DataProjectionSubstring(column_list, rename).apply(data)


# Selection means which rows are to be returned.

def data_selection_list(data=None, row_list=[]):
    """Extract a Sub-DataFrame selecting a specific row list.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param row_list: A list with the names of the rows to keep
    :type row_list: list

    :return: a Sub-DataFrame with selected rows
    :rtype: DataFrame
    """
    return pro.DataSelectionList(row_list).apply(data)


def data_selection_index(data=None, indx=0):
    """Extract a Sub-DataFrame selecting a specific row by index.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param indx: a number of a row. Must be on range of lenght row
    :type indx: int

    :return: a Sub-DataFrame with selected row
    :rtype: DataFrame
    """
    return pro.DataSelectionIndex(indx).apply(data)


def data_selection_name(data=None, name = ""):
    """Extract a Sub-DataFrame selecting a specific row by name.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param name: a name of a row. Must exists
    :type name: str

    :return: a Sub-DataFrame with selected row
    :rtype: DataFrame
    """
    return pro.DataSelectionName(name).apply(data)


def data_selection_range(data=None, ini_row = 0, fini_row = -1):
    """Extract a Sub-DataFrame selecting a range of rows
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param ini_row: a initial position to select. Must be on range of lenght rows
    :type ini_row: int
    :param fini_row: a final position to select. Must be on range of lenght rows
    :type fini_row: int

    :return: a Sub-DataFrame with the selected rows in an interval
    :rtype: DataFrame
    """
    return pro.DataSelectionRange(ini_row, fini_row).apply(data)


def data_selection_filter(data=None, filter_ = None):
    """Extract a Sub-DataFrame selecting rows by filter.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param filter_: a filter to do the selection by rows
    :type filter_: pd.Series, pd.DataFrame

    :return: a Sub-DataFrame after apply a filter by rows
    :rtype: DataFrame
    """
    return pro.DataSelectionFilter(filter_).apply(data)


def data_selection_series(series : ut.pd.Series, data=None):
    """Extract a Sub-DataFrame selecting a specific row by series.
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param series: A series with a structure to keep
    :type series: list

    :return: a Sub-DataFrame with selected rows
    :rtype: DataFrame
    """
    return pro.DataSelectionSeries(series).apply(data)



# =============================================================================
#                               FILTERING
# =============================================================================


def filter_by_index(data=None, index_cond = ""):
    """Extract a Sub-DataFrame selected by a filter of index condition.
        For multi-index cases:
        index_cond must have index type (it is returned after  do dataframe.index)
        It keeps only the rows that are in index_cond
    
    :param data: The current DataFrane to be process
    :type data: DataFrame
    :param index_cond: a condition as filter
    :type index_cond: str

    :return: a Sub-DataFrame that keeps only the rows that are in index_cond
    :rtype: DataFrame
    """
    return pro.FilterByIndex(index_cond).apply(data)


def filter_dictionary(data=None, list_var = []):
    """Given a Dictionary, select only elements on a passed list
        
    :param dict_: The current Dictionary to be process
    :type dict_: dict
    :param list_var: new Count Matrix
    :type list_var: list

    :return: A new dictionary with filtered elements
    :rtype: dict
    """
    return pro.FilterDictionary(list_var).apply(data)



# =============================================================================
#                             INTERSECTION
# =============================================================================


def list_intersection(list_ = [], sublist = []):
    """Obtains a new list with commom elements of two list or array
    
    :param list_: The primary list to be process
    :type list_: list
    :param sublist: secondary list to find commom elements
    :type sublist: list

    :return: A list which is the intersection between both input lists
    :rtype: list
    """
    return pro.ListIntersection(sublist).apply(list_)


def list_intersection_substring(list_ = [], substrings = []):
    """Obtains a new list with commom elements of two lists or array
        It is enough if only match partial name of element
    
    :param list_: The primary list to be process
    :type list_: list
    :param substrings: secondary list to find commom elements
    :type substrings: list
    :return: A list with founded common substring between two list
    :rtype: list
    """
    return pro.ListIntersectionSubString(substrings).apply(list_)



# =============================================================================
#                                ZIP
# =============================================================================



def list_zip(list_ = [], sublist = []):
    """Creates a new dictionary using one list
        for the keys, and other to its values
    
    :param list_: The current list to be process
    :type list_: list
    :param sublist: a list with values of the dictionary
    :type sublist: list

    :return: with the current list used as keys, creates and returns a dictionary with second list as values
    :rtype: dict
    """
    return pro.ListZip(sublist).apply(list_)


def data_zip(data=None, key_column = "", values_column = ""):
    """Creates a dictionary using information about two columns
        of a DataFrame
        
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param key_column: column name of a DataFrame used for the dictionary keys
    :type key_column: str
    :param values_column: column name of a DataFrame used for the dictionary values
    :type values_column: str

    :return: with both columns indicates, create and returns a dictionary
    :rtype: dict
    """
    return pro.DataZip(key_column, values_column).apply(data)



# =============================================================================
#                                RENAME
# =============================================================================



def rename_columname(data=None, pos_ini = 0, pos_fin = 0):
    """Rename all names of the columns triping the name
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param pos_ini: position of the first character
    :type pos_ini: int
    :param pos_fin: position of the last character
    :type pos_fin: int

    :return: A DataFrame with the column renamed with a substring of the inner column name
    :rtype: DataFrame
    """
    return pro.RenameColumname(pos_ini, pos_fin).apply(data)


def rename_index(data=None, new_name = ""):
    """Rename the name of the current index of DataFrame
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param new_name: name that will replace current one
    :type new_name: str

    :return: The same DataFrame with the index changed
    :rtype: DataFrame
    """
    return pro.RenameIndex(new_name).apply(data)



# =============================================================================
#                                DUPLICATES
# =============================================================================


def duplicates(data=None, axis = 0, keep = False):
    """Returns a DataFrame with the duplicates rows or columns of a input DataFrame
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool

    :return: A dataframe that only contains duplicated columns or rows
    :rtype: DataFrame
    """
    return pro.DataDuplicates(axis, keep).apply(data)



# =============================================================================
#                              INFORMATION
# =============================================================================


def describe(data=None, perc = ""):
    """Show a statistic description about a DataFrame
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param perc: The percentiles to include in the output
    :type perc: list, float

    :return: descriptive statistics that summarize the central tendency, dispersion and shape 
        of the dataset’s distribution, excluding NaN values
    :rtype: DataFrame
    """
    return pro.Describe(perc).apply(data)


def count_types(data=None, col_name = ""):
    """Shows a Series containing counts of unique values of a specific column.
        It will be in descending order so that the first element is the most frequently-occurring element
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param col_name: name of the column to count types
    :type col_name: str
    :return: information about types of a column
    :rtype: DataFrame
    """
    return pro.CountTypes(col_name).apply(data)



# =============================================================================
#                               REPLACE
# =============================================================================


def replace(data=None, to_replace = "Unknown", replaced_by = ut.np.nan):
    """Replace DataFrame's selected values with another.

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool

    :return: a DataFrame with all values indicates replaced by other value
    :rtype: DataFrame
    """
    return pro.Replace(to_replace, replaced_by).apply(data)


def fill_nan(data=None, to_replace = "Unknown", replaced_by = ut.np.nan):
    """Replace nan values to mean of the row of DataFrame
        Only modifies values of DataFrame.

    :param data: The current DataFrame to be process
    :type data: DataFrame

    :return: A DataFrame with all nan values replace to mean
    :rtype: DataFrame
    """
    return pro.FillNan(to_replace, replaced_by).apply(data)



# =============================================================================
#                              TRANSPOSE
# =============================================================================


def transpose(data=None, index_name = ""):
    """Do transpose to a DataFrame keeping the name of columns and index
        without multi-index
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param index_name: the name of the index before transpose to keep it
    :type index_name: str, int

    :return: a DataFrame with transpose of the input taking into account the extra row that it is created after the transpose
    :rtype: DataFrame
    """
    return pro.Transpose(index_name = index_name).apply(data)



# =============================================================================
#                                ADD
# =============================================================================


def add_column(data=None, name_column = "", values = []):
    """Add new column to a DataFrame
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of DataFrame
    :type values: list, ndarray

    :return: the input DataFrame with a new column with its values
    :rtype: DataFrame
    """
    return pro.AddColumn(name_column, values).apply(data)


def add_label(data, list_0s, column_observed, name_label = "label"):
    """Add a special column to a DataFrame.
        It will be used on Machine Learning as the Target Column of Classification
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param list_0s: List of values that consider as 0 on Label value
    :type list_0s: list, ndarray
    :param column_observed: Column of the DataFrame used to decide when value of target is 0 or 1
    :type column_observed: str
    :param name_label: values of the new column. Must have the same lenght as number of rows of DataFrame
    :type name_label: list, ndarray

    :return: a DataFrame with a new column that represents the labels of data to classify
    :rtype: DataFrame
    """
    return pro.AddLabel(list_0s, column_observed, name_label=name_label).apply(data)



# =============================================================================
#                                DELETE
# =============================================================================


def drop_duplicated(data=None, axis = 0, keep = False, by_name = False):
    """Drop data duplicates of a DataFrame
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool

    :return: A new dataframe without duplicates rows or columns
    :rtype: DataFrame
    """
    return pro.DropDuplicates(axis, keep, by_name).apply(data)


def drop_values(data=None, to_delete = ut.np.nan, axis = 0, method = "all"):
    """Drop values of a DataFrame

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param to_delete: value to be delete.
    :type to_delete: str, float, int, bool, list, ndarray, 
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param method: all to check all values are the same, any if a partial value match
    :type method: str

    :return: A new dataframe within this values
    :rtype: DataFrame
    """
    return pro.DropValues(to_delete, axis, method).apply(data)


def drop_nan_by_thresh(data=None, axis = 0, thresh = 0):
    """Drop DataFrame's values limited by Threshold.

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param thresh: threshold to evaluate element. If value less than threshold, that will be selected to drop
    :type thresh: float, int

    :return: A dataframe with drop element checking a threshold
    :rtype: DataFrame
    """
    return pro.DropNanByThresh(axis, thresh).apply(data)


def drop_rows_by_colname(data=None, col_name = "", to_delete = ut.np.nan):
    """Drop DataFrame's rows searching a value on a specific column

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param col_name: name of the column to search
    :type col_name: str
    :param to_delete: value to find on a column to decide with row delete is exists
    :type to_delete: str, float, int, bool, list, ndarray

    :return: A dataframe with dropped rows search a value on a specific column
    :rtype: DataFrame
    """
    return pro.DropRowsByColname(col_name, to_delete).apply(data)



"""
######################################################

                QUANTITATIVE ANALYSIS

######################################################
"""

# =============================================================================
#                              CORRELATION
# =============================================================================


def var_correlation(data=None):
    """Calculates correlation between variables of a DataFrame

    :param data: The current DataFrame to be process
    :type data: DataFrame

    :return: A dataframe with the correlation between the variables
        with values from 0.0 to 1.0, being 1.0 the highest value of correlation
    :rtype: DataFrame
    """
    return pro.VarCorrelation().apply(data)


def remove_correlation(data=None, thresh = 0.0):
    """Remove the elements that has high correlation

    :param data: The current DataFrame to be process
    :type data: DataFrame

    :return: A dataframe with high correlation removed
    :rtype: DataFrame
    """
    return pro.RemoveCorrelation(thresh).apply(data)



# =============================================================================
#                               VARIANCE
# =============================================================================


def variance(data):
    """Calculates the variance of a matrix

    :param data: The current DataFrame to be process
    :type data: DataFrame

    :return: A dictionary with the variance of each variable
    :rtype: dict
    """
    return pro.Variance().apply(data)



# =============================================================================
#                              SELECTION
# =============================================================================


def top_variables(data=None, n_var = 0):
    """Calculates the best n variables of a DataFrame

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param n_var: Number of variables selected as the best
    :type n_var: int

    :return: A dataframe with n_var rows that are the best
    :rtype: DataFrame
    """
    return pro.TopVariables(n_var).apply(data)


def remove_low_reads(data=None, thresh = 0, al_least = 2):
    """Calculates the best n variables of a DataFrame

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param thresh: threshold that indicates the limit to remove feature less than it
    :type thresh: float
    :param at_least: Indicates the minimum match to pass the selection
    :type at_least: int

    :return: A dataframe with removed features which have less then ‘threshold’ reads all columns
    :rtype: DataFrame
    """
    return pro.RemoveLowReads(thresh, al_least).apply(data)


def thresh_by_perc(data, perc = 10.0, axis = 0):
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
    return ut.thresh_by_perc(data, perc, axis)



# =============================================================================
#                              NORMALIZATION
# =============================================================================


def cpm(data=None, log_method = False, prior_count = 2):
    """Normalize values of DataFrame using CPM method
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param log_method: True if CPM will do with logaritmic alghoritm, False otherwise
    :type log_method: bool
    :param prior_count: If log is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size
    :type prior_count: int

    :return: a DataFrame to computed one of the CPM method
    :rtype: DataFrame
    """
    return pro.CPM(log_method = log_method, prior_count = prior_count).apply(data)



"""
######################################################

                    RNA PROCESS

######################################################
"""


def rna_sample_condition(data=None, sep = "-"):
    """Classify a given sample analyze its barcode

    :param data: The current str to be process
    :type brc_string: str
    :param sep: indicates with character is used to separate values
    :type sep: str

    :return: classification of the sample by its barcode passed on input
    :rtype: str
    """
    return pro.SampleCondition(sep).apply(data)


def rna_stat_test(data, grouped_by, group_1, group_2, clinical_data):
    """Generate a DataFrame with several statistic information like FoldChange, T, P-Value and a DiffExpr

    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param grouped_by: name of the column of clinical data to organize data
    :type grouped_by: str
    :param group_1: one of the type to create a group
    :type group_1: str
    :param group_2: another type to create a group
    :type group_2: str
    :param clinical_data: a dataframe with clinical information of the data
    :type clinical_data: DataFrame

    :return: A dataframe with several stadistic information: FoldChange, T, P-Value and a DiffExpr
    :rtype: DataFrame
    """
    return pro.StatTest(grouped_by, group_1, group_2, clinical_data).apply(data)


def rna_genes_lenght(genemodel, colnm_lenght = "width"):
    """Given a DataFrame with the genemodel information, return all lenght of the actual studied genes
    
    :param genemodel: The current DataFrame to be process
    :type genemodel: DataFrame
    :param colnm_lenght: name of the column that contains information about their lenght
    :type colnm_lenght: str

    :return: a dataframe with lenghts of the genes
    :rtype: DataFrame
    """
    return pro.GenesLenght(colnm_lenght).apply(genemodel)


def rna_gen_length(genemodel, gen):
    """Returns lenght of a specific gen indicates
    
    :param genemodel: a DataFrame with genemodel information
    :type genemodel: DataFrame
    :param gen_id: identifier of the gene on a genemodel
    :type gen_id: str

    :return: the lenght of a specific gen selected
    :rtype: int
    """
    return pro.GeneIDLenght(gen_id=gen).apply(genemodel)


def rna_rpk(data, genemodel):
    """Normalize values of DataFrame using RPK method
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame

    :return: a DataFrame with computed RPK
    :rtype: DataFrame
    """
    return pro.RPK(genemodel).apply(data)


def rna_tpm(data, genemodel):
    """Normalize values of DataFrame using TPM method
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame

    :return: a DataFrame with computed TPM
    :rtype: DataFrame
    """
    return pro.TPM(genemodel).apply(data)


def rna_rpkm(data, genemodel):
    """Normalize values of DataFrame using RPKM method
    
    :param data: The current DataFrame to be process
    :type data: DataFrame
    :param genemodel: a DataFrame with gene information, like gene lenght
    :type genemodel: DataFrame

    :return: a DataFrame with computed RPKM
    :rtype: DataFrame
    """
    return pro.RPKM(genemodel).apply(data)



"""
######################################################

                    DATA OBJECT

######################################################
"""


def create_data_object(counts_dframe : ut.pd.DataFrame, obs_ = ut.pd.DataFrame(), var_ = ut.pd.DataFrame(), log = Workflow(), unsdict = {} ):
    """Creates a Data Object to save information about a project downloaded or read

    :param counts_dframe: That will be the count matrix of the DataObject
    :type counts_dframe: DataFrame 
    :param obs_: That will be the clinical information associated to count matrix
        on DataObject
    :type obs_: DataFrame
    :param var_: That will be the gene information associated to count matrix
        on DataObject
    :type var_: DataFrame
    :param log: That will contains a flow that save step by step the changes of
        the DataObject. Useful to replicate an experiment with another information
    :type log: Workflow
    :param unsdict: It is a variable used to save extra data with an unclassified distribution
        on DataObject.
    :type unsdict: dict

    :return: A DataObject with all information organize
    :rtype: DataObject
    """
    return dobj.DataObject(counts_dframe = counts_dframe, obs_ = obs_, var_ = var_, log = log, unsdict = unsdict )



# =============================================================================
#                        BASIC FUNCTIONS FOR COUNTS
# =============================================================================


def set_counts(dobject : DataObject, new_counts = ut.pd.DataFrame()):
    """Set DataObject's Count Matrix to new one.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param new_counts: new Count Matrix
    :type new_counts: DataFrame
    
    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.SetCounts(new_counts).apply(dobject)


def update_counts(dobject : DataObject, new_counts = ut.pd.DataFrame()):
    """Set DataObject's Count Matrix to new one but only modify its values.
        In this case, the function not control that the rest of the data match with new Count Matrix
        because any row or column will be modify. Will have the same shape
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param new_counts: new Count Matrix
    :type new_counts: DataFrame

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.UpdateCounts(new_counts).apply(dobject)



# =============================================================================
#                       ACCESSING COUNTS DATAFRAME 
# =============================================================================


def counts_selection(dobject : DataObject, row_list = []):
    """Set DataObject's Count Matrix selecting a specific row list.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param row_list: A list with the names of the rows to keep
    :type row_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsSelection(row_list).apply(dobject)


def counts_projection(dobject : DataObject, col_list = []):
    """Set DataObject's Count Matrix selecting a specific column list.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param col_list: A list with the names of the columns to keep
    :type col_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsProjection(col_list).apply(dobject)


def counts_projection_substring(dobject : DataObject, col_list = [], rename = False):
    """Set DataObject's Count Matrix selecting a specific column list.
        In that case, is not neccesary that name of the column were exactly
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param col_list: A list with the names of the columns to keep
    :type col_list: list
    :param rename: True if the column names of Count Matrix need to be renamed, False otherwise
    :type rename: bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsProjectionSubstring(col_list, rename).apply(dobject)


def counts_replace_nan(dobject : DataObject):
    """Replace nan values to mean of the row of Count Matrix
        Only modifies values of Count Matrix.
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsReplaceNan().apply(dobject)


def counts_drop_duplicates(dobject : DataObject, axis = 0, keep = False, by_name = False):
    """Drop DataObject's Count Matrix duplicates.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Count Matrix.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool
   
    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsDropDuplicates(axis, keep, by_name).apply(dobject)


def counts_drop_values(dobject : DataObject, to_delete = 0, axis = 0, method = "all"):
    """Drop DataObject's Count Matrix values.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param to_delete: value to be delete.
    :type to_delete: str, float, int, bool, list, ndarray, 
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param method: all to check all values are the same, any if a partial value match
    :type method: str

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsDropValues(to_delete = to_delete, axis = axis, method = method).apply(dobject)


def counts_drop_nan_by_thresh(dobject : DataObject, axis = 0, thresh = 0):
    """Drop DataObject's Count Matrix values limited by Threshold.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param axis: Indicates 0 to row and 1 to columns
    :type axis: int
    :param thresh: threshold to evaluate element. If value less than threshold, that will be selected to drop
    :type thresh: float, int

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsDropNanByThresh(axis, thresh).apply(dobject)


def counts_replace(dobject : DataObject, to_replace = "Unknown", replaced_by = ut.np.nan):
    """Replace DataObject's Count Matrix values with another.
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsReplace(to_replace, replaced_by).apply(dobject)


def counts_transpose(dobject : DataObject, index_name = ""):
    """Do transpose to DataObject's Count Matrix
        Control that the rest of the data match with new Count Matrix
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param index_name: the name of the index before transpose to keep it
    :type index_name: str, int

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsTranspose(index_name = index_name).apply(dobject)


def counts_cpm(dobject : DataObject, log_method = False, prior_count = 2):
    """Normalize DataObject's Count Matrix using CPM method
        It is not neccesary to control that the rest of the data match with new Count Matrix.
        It only modify values inside Count Matrix.
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
        
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param log_method: True if CPM will do with logaritmic alghoritm, False otherwise
    :type log_method: bool
    :param prior_count: If log is True, ends up getting scaled by the ratio of a library size to the average library size and then multiplied by its value before getting added to each library size
    :type prior_count: int

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.CountsCPM(log_method = log_method, prior_count = prior_count).apply(dobject)



"""

OBSERVATIONS

"""


def set_obs(dobject : DataObject, new_obs = ut.pd.DataFrame()):
    """Set DataObject's Clinical Information to new one.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param new_obs: new Clinical Information
    :type new_obs: DataFrame

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.SetObs(new_obs).apply(dobject)



# =============================================================================
#                           ACCESSING OBS DATAFRAME 
# =============================================================================


def obs_selection(dobject : DataObject, row_list = []):
    """Set DataObject's Observations selecting a specific row list.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param row_list: A list with the names of the rows to keep
    :type row_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsSelection(row_list).apply(dobject)


def obs_projection(dobject : DataObject, col_list = []):
    """Set DataObject's Observations selecting a specific column list.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param col_list: A list with the names of the columns to keep
    :type col_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsProjection(col_list).apply(dobject)


def obs_add_column(dobject : DataObject, name_column = "", values = []):
    """Add to DataObject's Observations a new column
        It is not neccesary control that the rest of the data match with new Observations.
        Add new column doesn't affect the rest of the DataObject
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of Observations
    :type values: list, ndarray

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsAddColumn(name_column, values).apply(dobject)


def obs_replace(dobject : DataObject, to_replace = "Unknown", replaced_by = ut.np.nan):
    """Replace DataObject's Observations values with another.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsReplace(to_replace, replaced_by).apply(dobject)


def obs_drop_duplicates(dobject : DataObject, axis = 0, keep = False, by_name = False):
    """Drop DataObject's Observations duplicates.
        Control that the rest of the data match with new Observations
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Observations.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsDropDuplicates(axis, keep, by_name).apply(dobject)



"""

VARIABLES

"""


def set_vars(dobject : DataObject, new_var = ut.pd.DataFrame()):
    """Set DataObject's Variables to new one.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param new_var: new Gene Information
    :type new_var: DataFrame
    
    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.SetVars(new_var).apply(dobject)
    



# =============================================================================
#                           ACCESSING VAR DATAFRAME 
# =============================================================================


def vars_selection(dobject : DataObject, row_list = []):
    """Set DataObject's Variables selecting a specific row list.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param row_list: A list with the names of the rows to keep
    :type row_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.ObsSelection(row_list).apply(dobject)


def vars_projection(dobject : DataObject, col_list = []):
    """Set DataObject's Variables selecting a specific column list.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param col_list: A list with the names of the columns to keep
    :type col_list: list

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.VarsProjection(col_list).apply(dobject)


def vars_add_column(dobject : DataObject, name_column = "", values = []):
    """Add to DataObject's Variables a new column
        It is not neccesary control that the rest of the data match with new Variables.
        Add new column doesn't affect the rest of the DataObject
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param name_column: Name of the new column
    :type name_column: str
    :param values: values of the new column. Must have the same lenght as number of rows of Variables
    :type values: list, ndarray

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.VarsAddColumn(name_column, values).apply(dobject)


def vars_replace(dobject : DataObject, to_replace = "Unknown", replaced_by = ut.np.nan):
    """Replace DataObject's Variables values with another.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log

    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param to_replace: values to be replaced
    :type to_replace: str, float, int, bool, list, ndarray
    :param replaced_by: new value to replaced the value before
    :type replaced_by: str, float, int, bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.VarsReplace(to_replace, replaced_by).apply(dobject)


def vars_drop_duplicates(dobject : DataObject, axis = 0, keep = False, by_name = False):
    """Drop DataObject's Variables duplicates.
        Control that the rest of the data match with new Variables
        This class is a Task which allows generate a Workflow
        When this class is called, save his instance on DataObject's Log
    
    :param dobject: The current DataObject to be process
    :type dobject: DataObject
    :param axis: axis 0 means by rows, 1 by columns
    :type axis: int
    :param keep: Indicates which duplicates will keep on Observations.
        \"first\" keeps only the first duplicated column/row; \"last\" keeps only the last duplicated column/row; False keeps both duplicated columns/rows
    :type keep: str, bool
    :param by_name: True if only evaluate the name of the column/row, False if evaluate the content of the column/row to decide if is duplicated or not
    :type by_name: bool

    :return: The DataObject passed with the Task applied
    :rtype: DataObject
    """
    return dobj.VarsDropDuplicates(axis, keep, by_name).apply(dobject)



"""
######################################################

                        MODEL

######################################################
"""


def create_model_selection(data = None, source = None, target = None):
    """This class performs different techniques to the data such as: splitting data,
        over-sampling and under-sampling, normalization, computes confusion matrix and AUC, etc.,
        to the data in order to improve Machine Learning algorithm's performance.
    
    :param data: The data to apply Machine Learning methods. Defaults to None.
        If it is not specified, user must provide the `source` and `target` values.
    :type data: DataFrame
    :param source: The features that describe the samples. Defaults to None.
        If it is noy specified, user must provide `data` value. 
        Otherwise, if it is different to None, `target` values must be provided.
    :type source: DataFrame
    :param target: The labels to be predicted. Defaults to None.
        If it is noy specified, user must provide `data` value. 
        Otherwise, if it is different to None, `source` values must be provided.
    :type target: Series
    :param X_train: All the observations that will be used to train the model.
        Defaults to None. Only can be updated after use `get_train_test_sample` method.
    :type X_train: DataFrame
    :param Y_train: The dependent variable wich need to be predicted by the model.
        Defaults to None. Only can be updated after use `get_train_test_sample` method.
    :type Y_train: Series
    :param X_test:  The remaining portion of the independent  variables which 
        will not be used in the training phase and will be used
        to make predictions to test the accuracy of the model.
        Defaults to None. Only can be updated after use `get_train_test_sample` method.
    :type X_test: DataFrame
    :param Y_test: The labels of the test data. These labels will be used to test 
        the accuracy between actual and predicted categories. Defaults to None.
        Only can be updated after use `get_train_test_sample` method.
    :type Y_test: Series

    :return: a Model Selection with a data used on Machine Learning
    :rtype: ModelSelection
    """
    return ModelSelection(data = data, source = source, target = target)


def model_logistic_regression(name, penalty = 'l2', dual = False, tol = math.pow(10,-4),
                 C = 1.0, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
                 solver = 'lbfgs', max_iter = 100, multi_class = 'auto', verbose = 0, warm_start = False, 
                 n_jobs = None, l1_ratio = None):
    """Generate a Logistic Regression algorithm from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param penalty: Specify the norm of the penalty, where `none` means no penalty is added,
        `l2` uses Ridge regularization, `l1` uses Lasso regularization, and `elasticnet`
        means that both `l1` and `l2` penalty terms are added. Defaults to `l2`.
    :type penalty: `l1`,`l2`,`elasticnet`, `none`
    :param dual: Dual or primal formulation. Dual formulation is only implemented 
        for l2 penalty with liblinear solver. Defauls to `False`.
    :type dual: bool
    :param tol: Tolerance for stopping criteria. Defaults to `1e-4`.
    :type tol: float
    :param C: Inverse of regularization strenght, must be a positive float. 
        Smaller values specify stronger regularization. Defaults to `1.0`.
    :type C: float
    :param fit_intercept: Specifies if a constant (bias or intercept) should be added
        to the decision function. Defaults to `True`.
    :type fit_intercept: bool
    :param intercept_scaling: Useful only when the solver `liblinear` is used and
        `fit_intercept` True. In this case, x becomes [x, self.intercept_scaling].
        Defaults to `1`.
    :type intercept_scaling: float
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, `balanced`, or None
    :param random_state: Used when solver == `sag`, `saga` or `liblinear` to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None
    :param solver: Algorith to use in the optimization problem. Defaults to `lbfgs`.
    :type solver: {`newton-cg`, `lbfgs`, `liblinear`, `sag`, `saga`}
    :param max_iter: Maximum number of iterations taken for the solvers to converge.
        Defaults to `100`.
    :type max_iter: int
    :param multi_class: If the option chosen is `ovr`, then a binary problem is fit 
        for each label. For `multinomial` the loss minimised is the multinomial loss fit
        across the entire probability distribution, even when the data is binary.
        Defaults to `auto`.
    :type multi_class: {`auto` , `ovr` , `multinomial`}
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity. Defaults to ``0`.
    :type verbose: int
    :param warm_start: If `True`, reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to `False`.
    :type warm_start: bool
    :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class=`ovr`.
        Defaults to None.
    :type n_jobs: int
    :param l1_ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
        Only used if penalty = `elasticnetV. Defaults to None.
    :type l1_ratio: float

    :return: A Logistic Regression model initialize with arguments
    :rtype: LogisticRegression
    """
    
    return LogisticRegression(name = name, penalty = penalty, dual = dual, tol = tol,
                 C = C, fit_intercept = fit_intercept, intercept_scaling = intercept_scaling, class_weight = class_weight, random_state = random_state,
                 solver = solver, max_iter = max_iter, multi_class = multi_class, verbose = verbose, warm_start = warm_start, 
                 n_jobs = n_jobs, l1_ratio = l1_ratio)


def model_random_forest(name, n_estimators = 100, criterion = 'gini', max_depth = None, min_samples_split = 2,
               min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = 'sqrt', max_leaf_nodes = None, 
               min_impurity_decrease = 0.0, bootstrap = True, oob_score = False, n_jobs = None, random_state = None, verbose = 0,
               warm_start = False, class_weight = None, ccp_alpha = 0.0, max_samples = None):
    """Generate a Random Forest classifier algorithm from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param n_estimators: Number of trees in the forest. Defaults to `100`.
    :type n_estimators: int
    :param criterion: Function to measure the quality of a split. Defaults to `gini`.
    :type criterion: {`gini`, `entropy`, `log_loss`}.
    :param max_depth: The maximum depth of the tree. If None, nodes are expanded
        until all leaves are pure or contains less than `min_sample_split` samples.
        Defaults to None.
    :type max_depth: int, None
    :param min_samples_split: The minimum number of samples required to split an
        internal node. Defaults to `2`.
    :type min_samples_split: int or float
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        Defaults to `1`.
    :type min_samples_leaf: int or float
    :param min_weight_fraction_leaf: The minimum weighted fraction of the sum total
        of weights (of all the input samples) required to be at a leaf node.
        Defaults to `0.0`.
    :type min_weight_fraction_leaf: float
    :param max_features: The number of features to consider when looking for
        the best split. Defaults to `sqrt`.
    :type max_features:{`sqrt`, `log2`, None}, int or float
    :param max_leaf_nodes: Grow trees with `max_leaf_nodes` in best-first fashion.
        Defaults to None.
    :type max_leaf_nodes: int, None
    :param min_impurity_decrease: A node will be split if this split induces a decrease
        of the impurity greater than or equal to this value. Defaults to `0.0`.
    :type min_impurity_decrease: float
    :param bootstrap: Whether bootstrap samples are used when building trees. 
        If `False`, the whole dataset is used to build each tree. Defauls to `True`.
    :type bootstrap: bool
    :param oob_score: Wheter to use out-of-bag samples to estimate the generalization score.
        Only available if boostrap is `True`. Defaults to `False`.
    :type oob_score: bool
    :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class=`ovr`.
        Defaults to None.
    :type n_jobs: int
    :param random_state: Used when solver == `sag`, `saga` or `liblinear` to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity. Defaults to `0`.
    :type verbose: int
    :param warm_start: If `True`, reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to `False`.
    :type warm_start: bool
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, `balanced`, `balanced_subsample`, or None
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
        Defaults to `0.0`.
    :type ccp_alpha: non-negative float
    :param max_samples: If boostrap is `True`, the number of samples to draw from X to 
        train each base estimator. Defaults to None.
    :type max_samples: int, float, None

    :return: Random Forest classifier algorithm
    :rtype: RandomForest
    """
    return RandomForest(name = name, n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
               min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, 
               min_impurity_decrease = min_impurity_decrease, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose,
               warm_start = warm_start, class_weight = class_weight, ccp_alpha = ccp_alpha, max_samples = max_samples)


def model_scikit_learn(name, model = None):
    """Creates a generic model that is provided by
    the Scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param model: Model to work with.
    :type model: object

    :return: a generic ScikitLearn model
    :rtype: ScikitLearnModel
    """
    return ScikitLearnModel(name, model)


def model_support_vector_classif(name, C = 1.0, kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, 
               shrinking = True, probability = False, tol = math.pow(10,-3), cache_size = 200, class_weight = None, 
               verbose = False, max_iter = -1, decision_function_shape = 'ovr', break_ties = False, random_state = None):
    """Generate a Support Vector Machine algorithm from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param C: Regularization parameter. The strenght of the regularization is 
        inversely proportional to C. Defaults to `1.0`.
    :type C: float
    :param kernel: Specifies the kernel type to be used in the algorithm. 
        Defaults to `rbf`
    :type kernel: {`linear`, `poly`, `rbf`, `sigmoid`, `precomputed`} or callable
    :param degree: Degree of the polynomial kernel function when kernel is `poly`.
        Defaults to `3`.
    :type degree: int
    :param gamma: Kernek coefficient fot `rbf`, `poly` and `sigmoid`.
        Defaults to `scale`.
    :type gamma: {`scale`, `auto`} or float
    :param coef0: Independent term in kernel function, when kernel is `poly` or
        `sigmoid`. Defaults to `0.0`.
    :type coef0: float
    :param shrinking: Whether to use the shrinking heuristic. Defaults to `True`.
    :type shrinking: bool
    :param probability: Wheter to enable probability estimates. Deafuls to `False`.
    :type probability: bool
    :param tol: Tolerance for stopping criteria. Defaults to `1e-3`.
    :type tol: float
    :param cache_size: Specify the size of the kernelk cache (in MB). Defaults to `200`.
    :type cache_size: float
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, `balanced`, or None
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.  Defaults to None.
    :type verbose: bool
    :param max_iter: Hard limit on iterations within solver, or -1 for no limit.
        Defaults to -1.
    :type max_iter: int
    :param decision_function_shape: Wheter to return a one-vs-rest (`ovr`) or 
        one-vs-one (`ovo`) decision function. Defaults to `ovr`.
    :type decision_function_shape: {`ovo`, `ovr`}
    :param break_ties: If `True`, `decision_functoin_shape` = `ovr` and number of
        classes > 2, predict will break ties according to the confidence values
        of `decision_function`. Defaults to `False`.
    :type break_ties: bool
    :param random_state: Used when solver == `sag`, `saga` or `liblinear` to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None

    :return: a support vector machine algorithm
    :rtype: SupportVectorClassif
    """
    return SupportVectorClassif(name = name, C = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, 
               shrinking = shrinking, probability = probability, tol = tol, cache_size = cache_size, class_weight = class_weight, 
               verbose = verbose, max_iter = max_iter, decision_function_shape = decision_function_shape, break_ties = break_ties, random_state = random_state)


def model_neural_network(name, hidden_layer_sizes = (100,), activation = 'relu', solver = 'adam',
               alpha = 0.0001, batch_size = 'auto', learning_rate = 'constant', learning_rate_init = 0.001, power_t = 0.5,
               max_iter = 200, shuffle = True, random_state = None, tol = math.pow(10,-4), verbose = False, warm_start = False,
               momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, 
               beta_2 = 0.999, epsilon =  math.pow(10,-8), n_iter_no_change = math.pow(10,-8), max_fun = 15000):
    """Generates a Neural Network algorithm from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param hidden_layer_sizes: The ith element represents the number of neurons in the
        ith hidden layer. Defaults to `(100,)`.
    :type hidden_layer_sizes: tuple, length = n_layers - 2
    :param activation: Activation function for the hidden layer. Defaults to `relu`.
    :type activation: {`identity`, `logistic`, `tanh`, `relu`}
    :param solver: The solver for weight optimization. Defaults to `adam`.
    :type solver: {`lbfgs`, `sgd`, `adam`}
    :param alpha: Strength of the L2 regularization term (which is divided by the sample
        size when added to the loss). Defaults to `0.0001`.
    :type alpha: float
    :param batch_size: Size of minibatches for stochastic optimizers. Defaults to `auto`.
    :type batch_size: int, str
    :param learning_rate: Learning rate schedule for weight purposes. Defaults to `constant`.
    :type learning_rate: {`constant`, `invscaling`, `adaptive`}
    :param learning_rate_init: The initial learning rate used. It controls the step-size
        in updating the weights. Only when solver is `sgd` or `adam`. Defaults to `0.001`.
    :type learning_rate_init: float
    :param power_t: The exponent for inverse scaling learning rate. It is used in updating
        effective learning rate when the `learning_rate` is `invscaling`. Only when solver 
        is `sgd`. Defaults to `0.5`
    :type power_t: float
    :param max_iter: Maximum number of iterations. Defaults to `200`.
    :type max_iter: int
    :param shuffle: Whether to shuffle samples in each iteration. Only when solver is `sgd` 
        or `adam`. Defaults to `True`.
    :type shuffle: bool
    :param random_state: Determines random number generation for weights and bias initialization,
        train-test split and batch sampling. Defaults to None.
    :type random_state: int, RandomInstance, None
    :param tol:Tolerance for stopping criteria. Defaults to `1e-4`.
    :type tol: float
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.  Defaults to `False`.
    :type verbose: bool
    :param warm_start: If `True`, reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to `False`.
    :type warm_start: bool
    :param momentum: Momentum for gradient descent update. Should be between 0 and 1. 
        Defaults to `0.9`.
    :type momentum: float
    :param nesterovs_momentum: Wheter to use Nesterov's momentum. Only when solves is `sgd`
        and momentum > 0. Defaults to `True`.
    :type nesterovs_momentum: bool
    :param early_stopping: Whether to use early stopping to terminate training when 
        validation score is not improving. Defaults to `False`.
    :type early_stopping: bool
    :param validation_fraction: The proportion of training data to set aside as validation 
        set fot early stopping. Must be between 0 and 1. Only if early stopping is `True`.
        Defaults to `0.1`.
    :type validation_fraction: float
    :param beta_1: Exponential decay rate for estimates of first moment vector in adam, 
        should be between 0 and 1. Only used when solver is `adam`. Defaults to `0.9`.
    :type beta_1: float
    :param beta_2: Exponential decay rate for estimates of second moment vector in adam, 
        should be between 0 and 1. Only used when solver is `adam`. Defaults to `0.999`.
    :type beta_2: float
    :param epsilon: Value for numerical stability in adam. Only used when solver is `adam`. 
        Defaults to `1e-8`.
    :type epsilon: float
    :param n_iter_no_change: Maximum number of epoch to not meet tol improvement. 
        Only when solver is `sgd` or `adam`. Defaults to `10`.
    :type n_iter_no_change: int
    :param max_fun: Maximum number of loss functions calls. Only when solver is `lbfgs`. 
        Defaults to `15000`.
    :type max_fun: int

    :return: Neural Network algorithm
    :rtype: NeuralNetwork
    """
    return NeuralNetwork(name = name, hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver,
               alpha = alpha, batch_size = batch_size, learning_rate = learning_rate, learning_rate_init = learning_rate_init, power_t = power_t,
               max_iter = max_iter, shuffle = shuffle, random_state = random_state, tol = tol, verbose = verbose, warm_start = warm_start,
               momentum = momentum, nesterovs_momentum = nesterovs_momentum, early_stopping = early_stopping, validation_fraction = validation_fraction, beta_1 = beta_1, 
               beta_2 = beta_2, epsilon = epsilon, n_iter_no_change = n_iter_no_change, max_fun = max_fun)



"""
######################################################

                    PROCESSGDC

######################################################
"""


# =============================================================================
#                                   URL 
# =============================================================================


def gdc_endpt_base():
    """Returns a string with the URL to search in the TCGA base
        
    :return: The URL to search in the GDC base.
    :rtype: str
    """
    return ProcessGDC.get_endpt_base()


def gdc_endpt_files():
    """Returns a string with the URL to files endpoint
        
    :return: The URL to search in the GDC files.
    :rtype: str
    """
    return ProcessGDC.get_endpt_files()


def gdc_endpt_status():
    """Returns a string with the URL to check status
        
    :return: The URL to check status
    :rtype: str
    """
    return ProcessGDC.get_endpt_status()


def gdc_endpt_data():
    """Returns a string with the URL to files endpoint
        
    :return: The URL to search in the GDC files.
    :rtype: str
    """
    return ProcessGDC.get_endpt_data()


def gdc_endpt_projects():
    """Returns a string with the URL to projects endpoint
        
    :return: The URL to search in the GDC projects.
    :rtype: str
    """
    return ProcessGDC.get_endpt_projects()


def gdc_endpt_legacy():
    """Returns a string with the URL to search in the TCGA Legacy Archive
        
    :return: The URL to search in the GDC Legacy Archive.
    :rtype: str
    """
    return ProcessGDC.get_endpt_legacy()


def gdc_clinical_filename():
    """Returns a string with the main name of a clinical filename of GDC
        
    :return: a string with the main name.
    :rtype: str
    """
    return ProcessGDC.get_clinical_filename()


def gdc_clinical_options():
    """Returns a list of options for clinical information
        
    :return: list of options
    :rtype: str
    """
    return ProcessGDC.get_clinical_options()


def gdc_data_legacy():
    """ Returns a boolean that indicates if the data is from the GDC
        Legacy Archive or from the GDC Data Portal.
    
        :return: `True` to search for data in the GDC Legacy Archive,
            and `False` otherwise.
        :rtype: bool
    """
    return ProcessGDC.get_data_legacy()


def gdc_set_data_legacy(legacy):
    """Set if data comes from the GDC Legacy Archive or not.
        
        :param legacy: `True` to search for data in the GDC Legacy Archive,
            and `False` otherwise.
        :type legacy: bool 
    """
    ProcessGDC.get_enset_data_legacydpt_legacy(legacy)



# =============================================================================
#                               PROJECTS 
# =============================================================================



def gdc_get_projects():
    """Gets all available GDC projects
            
        :return: List of available projects in the GDC Data Portal.
        :rtype: list
    """
    return ProcessGDC.get_projects()


def gdc_search_project(ident):
    """Search the project by the input "ident" variable. It could be the identifier
        or the complete name of the cancer. If exists, the ID will be returned.
        
        :param ident: Identificator (ID/Name) to search the project.
        :type ident: str
        :return: The identifier of the project. E.g: TCGA-BRCA.
        :rtype: str
    """
    return ProcessGDC.search_project(ident)


def gdc_check_parameter(endpt, param, value, filters):
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
    return ProcessGDC.check_parameter(endpt, param, value, filters)


def gdc_check_data_type(endpt, filters, data_type):
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
    return ProcessGDC.check_data_type(endpt, filters, data_type)


def gdc_form_filter(filters, nameFilter, valueFilter, fullNameFilter, endpt_):
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
    return ProcessGDC.form_filter(filters, nameFilter, valueFilter, fullNameFilter, endpt_)


def gdc_regex_file_name(proj_id, clinical_option, file_name):
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
        :return: `True` if the clinical option has been found and `False` otherwise.
        :rtype: bool
    """
    return ProcessGDC.regex_file_name(proj_id, clinical_option, file_name)


def gdc_find_clinical(file_list, clinical_option):
    """ Finds if the selected clinical option is one of the available list 
        of files.
        First, we only kept with those files from the list of files that match
        a pattern where it is searched tyhe word "clinical" in the name of the file.
        Then, in a new list with only the clinical files, search for a file that have
        the specified clinical option.
        
        :param file_list: List of files to search the clinical option's file.
        :type file_list: list
        :return: `True` if the clinical option has been found and `False` otherwise.
        :rtype: bool
    """
    return ProcessGDC.find_clinical(file_list, clinical_option)


def gdc_write_to_dataframe(list_, file, col, entity_id, legacy):
    """Creates a list of the DataFrame's lines, so it allows multithreading
        while reading the file.
        
        :param list_: List to append DataFrame's content. It is modified by reference.
        :type list_: list
        :param file: file to be read.
        :type file: str
        :param col: Name of the column to be read. If legacy is `True`, 
            it is set to the second column of the DataFrame; if legacy is
            `False`, column is set to `unstranded`.
        :type col: str
        :param legacy:`True` to search for data in the GDC Legacy Archive,
            and `False` otherwise; defaults to `False`.
        :type legacy: bool 
    """
    return ProcessGDC.write_to_dataframe(list_, file, col, entity_id, legacy)



# =============================================================================
#                               PROJECTS 
# =============================================================================



def gdc_query(query_object):
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
        :rtype: str
    """
    return ProcessGDC.get_query(query_object)


def gdc_clinical_query(query_object):
    """Query in the TCGA API to get the clinical info linked to a 
        specific project.
        
        :param project_name: Name of the project.
        :type project_name: str
        :param legacy:`True` to search for data in the GDC Legacy Archive,
            and `False` otherwise; defaults to `False`.
        :type legacy: bool 
    """
    return ProcessGDC.get_clinical_query(query_object)



# =============================================================================
#                               PROJECTS 
# =============================================================================



def gdc_download_data(query_object, os_path = None):
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

    return ProcessGDC.download_data(query_object, os_path)


def gdc_download_clinical_data(project_name, clinic_info, legacy = False, os_path = None):
    """ Download clinical data from a specified project. A folder with the name of the
        project will be created if it does not already exist. If it exists, it will be
        checked if the clinical information has been previously downloaded.
        
        :param project_name: Name of the project.
        :type project_name: str
        :param clinic_info: Type of clinical information that will be downloaded. 
            It will be downloaded the latest version found. E.g: "patient" or "drug".
        :type clinic_info: str
        :param legacy:`True` to search for data in the GDC Legacy Archive,
            and `False` otherwise; defaults to `False`.
        :type legacy: bool
        :param os_path: Directory path were the file will be saved. Defaults to None, so
            it will be saved in the current directory.
        :type os_path: str, optional
    """

    return ProcessGDC.download_clinical_data(project_name, clinic_info, legacy = legacy, os_path = os_path)


def gdc_download_genecode(url_gtf = "https://api.gdc.cancer.gov/data/be002a2c-3b27-43f3-9e0f-fd47db92a6b5", name_gtf = "genecode_v36"):
    """ Download genecode to create a data structure with information about
        specific genes. It will allow building normalization functions.

        :param url_gtf: URL where the genecode is hosted. By default, the program download from GDC and its 36th version.
        :type url_gtf: str
        :param name_gft: Name of the file (WITHOUT extension). Its extension must be .gtf
        :param name_gft: str
    """

    ProcessGDC.download_genecode(url_gtf, name_gtf)



def gdc_create_gene_model(path_gtf = "genecode_v36.gtf", path_save = "genecode_v36.csv"):
    """Create a DataFrame that allow an easy treatment with information about Genes.
        The function will create a CSV readable with features of Genes. If CSV exist, there is
        not neccesary to call this function, only read it.

        :param path_gtf: Path where genecode with GTF format is located
        :type path_gtf: str
        :param path_save: Path where the CSV result will be save
        :param path_save: str
    """

    ProcessGDC.create_gene_model(path_gtf, path_save)



def read_genecode_csv(file_csv="../genecode_v36.csv", sep=","):
    """Read a CSV file that contains information about genes
        and returns a DataFrame to work with it

        :param file_csv: Path where the genecode on CSV format is saved
        :type file_csv: str
        :param sep: Character that delimit its structure
        :type sep: str

        :return: A DataFrame with Gene Information
        :rtype: DataFrame
    """

    return ProcessGDC.read_genecode_csv(file_csv, sep)


# =============================================================================
#                           READ DATA FUNCTIONS 
# =============================================================================



def gdc_read_file_rna_legacy(file_name, col, sep = "\t"):
    """ Read a single Rna-Seq file from The TCGA Legacy Archive.
        The first line is the column's names.
        An undefined number of rows are skipped while their first character
        is a `?`.
        The index is set to the first column, which is the gene_id.
        In this type of files the gene_id are made up of the ID of the gene and a number,
        separated by a "|". Only the ID of the gene is saved.
        
        :param file_name: File name to be read.
        :type dir_path: str
        :param col: Name of the column to be read. Set to `unstranded`.
        :type col: str
        :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
        :type sep: str
        :return: DataFrame with file's data read.
        :rtype: DataFrame
    """

    return ProcessGDC.gdc_download_clinical_data(file_name, col, sep)


def gdc_read_file_rna(file_name, col, sep = "\t"):
    """ Read a single Rna-Seq file from The GDC portal. 
        The first line of this type of files is a comment that it is skipped.
        The second line will be the header of the data.
        Next lines are skipped / saved in a summary_file because they store 
        summary information of the file.
        The index is set to the first column, which is the gene_id.
        
        :param file_name: File name to be read.
        :type dir_path: str
        :param col: Name of the column to be read. Set to `unstranded`.
        :type col: str
        :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
        :type sep: str
        :return: DataFrame with file's data read.
        :rtype: DataFrame
    """

    return ProcessGDC.read_file_rna(file_name, col, sep)



def gdc_read_rna(dir_path, sep = "\t", save = True):
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
        :param save: `True` to keep the downloaded data in the folder and `False` to remove
        the downloaded data after read it; defaults to `True`.
        :type save: bool
        :return: DataFrame with the RNA-Seq information read.
        :rtype: DataFrame
    """

    return ProcessGDC.read_rna(dir_path, sep, save)


def gdc_read_clinical(file_name, sep="\t"):
    """ Read a clinical file preivously downaloaded from The Cancer Genome Atlas (TCGA) portal.
        In TCGA clinical files, the first line is a comment that it is skipped. The two following lines
        have the header names; only the second header's line is kept, due to they have the same values
        but with different names.
        Returns a dataframe with the clinical information where the index is set to `bcr_patient_barcode`,
        that match with the Entity ID of the samples.
        
        :param file_name: The name of the clinical project to read. By default, the programn search the file
            in the actual path, so the location of the file is required to find the file. 
            E.g: "TCGA-BRCA/nationwidechildrens.org_clinical_patient_brca.txt".
        :type file_name: str
        :param sep: Separator character between fields. Defaults to "\t" (Tab-separated values).
        :type sep: str
        :return: DataFrame with the clinical information read. The index has been set to
            `bcr_patient_barcode`, matching with the Entity ID of the samples.
        :rtype: DataFrame
    """

    return ProcessGDC.read_clinical(file_name, sep)



"""
######################################################

                        OBJECTS

######################################################
"""


# =============================================================================
#                               TCGAQUERY 
# =============================================================================


def create_gdc_query(project ,experimental_strategy, data_category = None, data_type = None, 
                 workflow_type = None, legacy = False,  platform = None, 
                 data_format = None, sample_type = None, normalized = False):
    """Generate a object to do arequest on Genomic Data Commoms to download data
    
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
        harmonized by the GDC. Legacy is `True` to search for data in the GDC Legacy Archive,
        and `False` otherwise; defaults to `False`.
    :type legacy: bool
    :param platform: Technolofical platform on which experimental data was produced.
        E.g: "Illumina Human Methylation 450", defaults to None. 
    :type platform: str, optional
    :param data_format: Format of the Data File, defaults to None.
    :type data_format: str, optional
    :param sample_type: Describes the source of a biospecimen used for a laboratory test. 
        E.g: "Primary Tumor", ["Primary Tumor", "Solid Tissue Normal"], defaults to None.
    :type sample_type: list, str, optional
    :param normalized: This parameter is only valid when legacy is set as `True`.
        If `True`, normalized data from the GDC Legacy Archive will be downloaded, 
        and `False` otherwise; defaults to `False`.
    :type normalized: bool

    :return: a object with all neccesary information to download the project
    :rtype: GDCQuery
    """
    return GDCQuery(project ,experimental_strategy, data_category, data_type, 
                 workflow_type, legacy,  platform, 
                 data_format, sample_type, normalized)



# =============================================================================
#                               DATAPROJECT 
# =============================================================================



def create_data_project():
    """Generate a object that allow to prepare a download for a project.
    Essential information such as the project's name, the path where it has been downloaded,
    the list of related files and the total size of the project is saved.
    
    :return: a object with all information save on its attributes
    :rtype: DataProject
    """
    return DataProject()



# =============================================================================
#                               FILEPROJECT 
# =============================================================================



def create_file_project(id_, data_format = None, access = None, file_name = None, submitter_id = None,
               data_category = None, type_ = None, file_size = None, created_datetime = None, md5sum = None,
               updated_datetime = None, data_type = None, state= None,  experimental_strategy = None, version = None,
               data_release = None, entity_id = None):
    """Creates a object with the information and features of a downloaded file. 
        Only the id_ will be a mandatory field due to the differences
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

    :return: a object with information of a specific File
    :rtype: FileProject
    """
    return FileProject(id_, data_format, access, file_name, submitter_id,
               data_category, type_, file_size, created_datetime, md5sum,
               updated_datetime, data_type, state,  experimental_strategy, version,
               data_release, entity_id)



# =============================================================================
#                                WORKFLOW 
# =============================================================================



def create_workflow():
    """Creates a object workflow that allows to save several Log of a DataObject
        The methods executed over the data can be saved and applied over bulk data 
        to replicate/expand the same analysis in the future.
    
    :return: an empty object prepare to save a Workflow of DataObject
    :rtype: Workflow
    """
    return Workflow()



"""
######################################################

                    SERVER STATUS

######################################################
"""

def server_status_code(stat_code):
    """Return the status response from the server
                
        :param stat_code: Current status of the checked URL endpoint.
        :type stat_code: int
        
        :return: Message explaining the obtained server's code.
        :rtype: str
    """

    return ut.status_code(stat_code)


def server_check_status(endpt_status):
    """Check the status of the server.
            
        :param endpt_status: URL (endpoint) to be checked.
        :type endpt_status: str
    """
    
    return ut.check_server_status(endpt_status)


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
    
    return ut.get_json_query(url_query, params_query)


def create_opin(filt):
    """Create a single in-operator in json format for the query, adding a field and its value.

        :param filt: values of a filter.
        :type filt: list[str]
        
        :return: the structured in-operator
        :rtype: dict
    """

    return ut.create_opin(filt)


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

    return ut.find_parameter(value, param_results, param)


def print_request(json_obj):
    """ Decodes a JSON object and display it in an elegant format per screen,

        :param json_obj: JSON object to print.
        :type json_obj: JSON Object
    """

    ut.print_request(json_obj)



"""
######################################################

                    UTILITIES

######################################################
"""


def check_have_header(filename):
    """Check if the file has or not header.
            
        :param filename: Name of the file to check if has header or not.
        :type filename: str

        :return: `True` if file has header, `False` if otherwise.
        :rtype: bool
    """

    return ut.check_have_header(filename)


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

    return ut.read_file(file, index, sep)


def copy_object(obj):
    """Comparison between objects doesn´t make a copy, 
        but just add another reference. 
        This method do a copy of an object.

        :param obj: Object to do the copy
        :type obj: Object

        :return: A copy of the object
        :rtype: Object
    """

    return ut.copy_object(obj)


def recursive_delete_data(dir_to_delete):
    """Search in the directory tree the input directory, so it remove its
        content recursively before remove the root directory.

        :param dir_to_delete: Directory path to delete its content.
        :type dir_to_delete: str
    """

    ut.recursive_delete_data(dir_to_delete)


def calc_plot_dim(n_elems):
    """Calculate the number of rows and columns of a given number
        in order to find the most squared and symetric image that it is possible

        :param n_elems: The number of elements.
        :type n_elems: int
    
        :return: The number of rows and columns.
        :rtype: int, int
    """

    return ut.calc_plot_dim(n_elems)


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

    ut.print_progress_bar(iteration, total, mb_iter, mb_total, prefix, suffix, length, decimals,  fill, printEnd)



"""
######################################################

                    READ JSON-DICT

######################################################
"""


def dict_from_dataobject(dobject):
    """Generates a Dictionary from DataObject information

    :param dobject: a object of class DataObject to extract information
    :type dobject: DataObject

    :return: a dictionary with the same information as input
    :rtype: dict
    """
    return ut.dict_from_dataobject(dobject)


def json_from_dict(data):
    """Generate a Json from dictionary information

    :param data: a dictionary to extract information
    :type data: dict

    :return: a str with json syntax with the same information as input
    :rtype: str
    """
    return ut.json_from_dict(data)


def dict_from_json(json):
    """Serialize a dictionary from json information

    :param json_data: a json structure to extract information
    :type json_data: str

    :return: a dictionary with the same information as input
    :rtype: dict
    """
    return ut.dict_from_json(json)


def dict_from_json_file(path):
    """Serialize a dictionary from json file

    :param path: a path with the json file to extract information
    :type path: str

    :return: a dictionary with the same information as input
    :rtype: dict
    """
    return ut.dict_from_json_file(path)


def json_file_from_dict(data, path = None):
    """Generate a Json file from dictionary information

    :param data: a dictionary to extract information
    :type data: dict
    :param path: path to save new file created. If None, it creates a file on the same
        path as program with the generic name 'object.json'
    :type path: dict
    """
    return ut.json_file_from_dict(data, path)