# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

from src import utils as ut
from src import processing as pro

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import webbrowser
import pathlib
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import zscore

# Set notebook mode to work in offline
pio.renderers.default='svg'

# R OBJECT
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

"""
utils = importr("utils")
utils.install_packages("BiocManager")

BiocManager = importr("BiocManager") 
BiocManager.install("limma") 

limma = importr("limma")
"""

###############################################################################
        
#                               GENERAL METHODS

###############################################################################
        
def update_layout(fig, xlabel, ylabel, x_ini_range, x_fin_range, y_ini_range, y_fin_range,
                  legend, legend_title, vheight, vwidth, title):
    """Updates the features of the figure to display.

    :param fig: Figure to update.
    :type fig: Figure
    :param xlabel: Name of the X axis.
    :type xlabel: str
    :param ylabel: Name of the Y axis.
    :type ylabel: str
    :param x_ini_range: Initial value of X axis range.
    :type x_ini_range: int
    :param x_fin_range: Final value of X axis range.
    :type x_fin_range: int
    :param y_ini_range: Initial value of Y axis range.
    :type y_ini_range: int
    :param y_fin_range: Final value of Y axis range.
    :type y_fin_range: int
    :param legend: Indicates if the figure will have legend or not.
    :type legend: bool
    :param legend_title: Name of the legend's label
    :type legend_title: str
    :param vheight: Height of the figure as image.
    :type vheight: int
    :param vwidth: Width of the figure as image.
    :type vwidth: int
    :param title: Title of the figure.
    :type title: str

    :return: The figure with the layout's features updated.
    :rtype: Figure
    """
    
    title = "<i><b>" + title + "</b></i>"
    
    fig.update_layout(title_text = title, 
                xaxis_title = xlabel,
                yaxis_title = ylabel,
                height = vheight, 
                width = vwidth,
                title_x = 0.5,
                legend_title_text=legend_title,
                showlegend = legend)


    if x_ini_range != None and x_fin_range != None:
        fig.update_xaxes(range=[x_ini_range, x_fin_range])
        
    if y_ini_range != None and y_fin_range != None:
        
        fig.update_yaxes(range=[y_ini_range, y_fin_range])
    
    return fig


def show_figure(fig, xlabel = "x axis", ylabel = "y axis", x_ini_range = None, x_fin_range = None,
                y_ini_range = None, y_fin_range = None, legend = True, legend_title = "",
                vheight = 600, vwidth = 800, title = "Plot of the figure"):
    """Show in the screen the figure plot with the specified figured parameters.

    :param fig: Figure to update.
    :type fig: Figure
    :param xlabel: Name of the X axis. Defaults to 'x axis'.
    :type xlabel: str
    :param ylabel: Name of the Y axis. Defaults to 'y axis'.
    :type ylabel: str
    :param x_ini_range: Initial value of X axis range. Defaults to None.
    :type x_ini_range: int
    :param x_fin_range: Final value of X axis range. Defaults to None.
    :type x_fin_range: int
    :param y_ini_range: Initial value of Y axis range. Defaults to None.
    :type y_ini_range: int
    :param y_fin_range: Final value of Y axis range. Defaults to None.
    :type y_fin_range: int
    :param legend: Indicates if the figure will have legend or not. Defaults to 'True'.
    :type legend: bool
    :param legend_title: Name of the legend's label
    :type legend_title: str
    :param vheight: Height of the figure as image. Defaults to 600.
    :type vheight: int
    :param vwidth: Width of the figure as image. Defaults to 800.
    :type vwidth: int
    :param title: Title of the figure. Defaults to 'Plot of the figure'.
    :type title: str
    """

    fig = update_layout(fig, xlabel = xlabel, ylabel = ylabel, x_ini_range = x_ini_range, x_fin_range = x_fin_range, y_ini_range = y_ini_range, 
                y_fin_range = y_fin_range, legend = legend, legend_title = legend_title, vheight = vheight, vwidth = vwidth, title = title)

    fig.show()


def save_image(fig, fig_name = "fig", img_format = "png", os_path = "images", 
               xlabel = "x axis", ylabel = "y axis", x_ini_range = None, x_fin_range = None,
               y_ini_range = None, y_fin_range = None, legend = True, legend_title = "", vheight = 600, vwidth = 800, 
               title = "Plot of the figure" ):
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
    :param xlabel: Name of the X axis. Defaults to 'x axis'.
    :type xlabel: str
    :param ylabel: Name of the Y axis. Defaults to 'y axis'.
    :type ylabel: str
    :param x_ini_range: Initial value of X axis range. Defaults to None.
    :type x_ini_range: int
    :param x_fin_range: Final value of X axis range. Defaults to None.
    :type x_fin_range: int
    :param y_ini_range: Initial value of Y axis range. Defaults to None.
    :type y_ini_range: int
    :param y_fin_range: Final value of Y axis range. Defaults to None.
    :type y_fin_range: int
    :param legend: Indicates if the figure will have legend or not. Defaults to 'True'.
    :type legend: bool
    :param legend_title: Name of the legend's label
    :type legend_title: str
    :param vheight: Height of the figure as image. Defaults to 600.
    :type vheight: int
    :param vwidth: Width of the figure as image. Defaults to 800.
    :type vwidth: int
    :param title: Title of the figure. Defaults to 'Plot of the figure'.
    :type title: str
    """
    
    valid_formats = ["png", "jpeg", "webp", "svg", "pdf"]
    
    if img_format in valid_formats:
        
        if not ut.os.path.exists(os_path):
            
            ut.os.mkdir(os_path)
            
        fig = update_layout(fig, xlabel = xlabel, ylabel = ylabel, x_ini_range = x_ini_range, x_fin_range = x_fin_range, y_ini_range = y_ini_range, 
                y_fin_range = y_fin_range, legend = legend, legend_title = legend_title, vheight = vheight, vwidth = vwidth, title = title)
            
        img_path = os_path + "/" + fig_name + "." + img_format
            
        try:
            fig.write_image(img_path)
            
        except:
            print("Unable to save the image.")
            return -1
    else:
    
        print("\nFormat type " + str(img_format) + " is not available. The format options availables are:\n")

        print("\t" + " ".join(valid_formats))
            
        ut.sys.exit(0)



def show_image_web(fig):
    """Show an image on a web browser. If there is not an available web browser,
    it shows the image on the default application to visualize images.

    :param fig: Figure to dislay.
    :type fig: Figure
    """
    
    # It is necessary to have the whole path of the file because the browser
    # won´t execute a relative path, but it'll do it from the path were it is
    # installed, so it is NEEDED the absolute path
    if ut.os.path.exists(fig):

        fullpath_img = str(pathlib.Path(fig).parent.resolve()) + "/" + str(fig.split(sep="/")[-1])
        
        # Local path of File type to be able to open it in the browser
        webname = "file://" + fullpath_img
        
        # If the browser is not installled it will raise an error so it will be used
        # the default application of the system
        # Try with the most used browsers: google chrome, mozilla firefox, opera and safari (MAC)
        try:
            webbrowser.get('google-chrome').open(webname)
        except:
            
            try: 
                webbrowser.get('mozilla').open(webname)
            except:
                
                try:
                    webbrowser.get('opera').open(webname)
                except:
                    
                    try:
                        webbrowser.get('safari').open(webname)
                        
                    except:
                        webbrowser.open(webname)

    else:

        print("File: " + str(fig) + " does not exist")



###############################################################################
        
#                      PREPROCESSING VISUALIZATIONS

###############################################################################     


# Displays a bar plot
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
    
    colsum = data.sum(axis = 0)
    
    fig = px.bar(colsum, x = x, y = y)
    
    return fig


def kde_plot(data):
    """Creates a Kernel Density Estimation (kde) diagram with the data received as the input.
    A kernel density estimate (KDE) plot is a method for visualizing the distribution 
    of observations in a dataset, analagous to a histogram.

    :param data: Data values to create the diagram.
    :type data: DataFrame

    :return: Figure that contains the KDE plot.
    :rtype: Figure
    """
    
    if isinstance(data, ut.pd.DataFrame):
        
        fig = ff.create_distplot([data[c] for c in data.columns], data.columns, show_hist=False, show_rug=False)
        
        return fig
        
    else:
        print("Input data must be a dataframe")
        ut.sys.exit(0)
        

def box_plot(data, x = None, y= None, median = False):
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
        Defaults to 'False'.
    :type median: bool

    :return: Figure that contains the box plot.
    :rtype: Figure
    """
    
    # If x and y is not None, they will be names of data columns
    if x == None and y == None:
        fig = px.box(data) 
        
    else:
        fig = px.box(data, x = x, y = y, color = "label")            
    
    if median == True:
        median = ut.np.median(data)
        fig.add_hline(y=median)
            
    return fig


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
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = x,
            y = y,
            z = z,
            colorscale = color
        )
    )
    
    return fig
    

# https://plotly.com/python/dendrogram/
# https://community.plotly.com/t/heatmap-with-dendrogam-and-y-ticks-labels/31694
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

    transpose = ut.copy_object(data)

    transpose = pro.Transpose().apply(transpose)

    new_datas_transpose = zscore(transpose.values, ddof = 0, nan_policy = "propagate")


    new_datas = ut.copy_object(new_datas_transpose.T)


    # (x - media) / sd
    #new_datas = (flipped.values - flipped.values.mean()) / flipped.values.std()    

    ##### ESTE ES EL DENDOGRAMA DE ARRIBA (MUESTRAS)

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(new_datas_transpose, orientation='bottom', labels=list(data.columns.values))
    
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'
    

    ###### ESTE ES EL DENDOGRAMA DE LA IZQUIERDA (GENES)

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(new_datas, orientation='right',labels=list(data.index.values))
        
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'
    
    # Add Side Dendrogram Data to Figure
    for d in dendro_side['data']:
        fig.add_trace(d)


    # Create Heatmap
    dendro_leaves = list(fig['layout']['xaxis']['ticktext'])
            
    dendro_insides = dendro_side['layout']['yaxis']['ticktext']

    dendro_insides = list(map(str, dendro_insides))
           
    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_insides,
            z = new_datas,
            colorscale = color
        )
    ]

    # zscore(data.values, ddof = 0, nan_policy = "propagate"),
    
    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)
    
    # Edit Layout
    fig.update_layout({'width':800, 'height':800,
                             'showlegend':False, 'hovermode': 'closest',
                             })

    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks':""})

    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""
                            })

    # Edit yaxis2
    fig.update_layout(yaxis2={'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""})
    
    return fig



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

    def as_dict_python(vector):
        """Convert an RPy2 ListVector to a Python dict"""
        result = {}
        for i, name in enumerate(vector.names):
            if isinstance(vector[i], ro.ListVector):
                result[name] = as_dict_python(vector[i])
            elif len(vector[i]) == 1:
                result[name] = vector[i][0]
            else:
                result[name] = vector[i]
        return result

    limma = importr("limma")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(data)

    mds = limma.plotMDS(r_from_pd_df) # ListVector

    dict_samples = as_dict_python(mds)

    x = dict_samples["x"]
    y = dict_samples["y"]

    colors_used = list(data.columns.values)

    if len(clinical_info) > 0:
        
        if len(x) == len(clinical_info):
            colors_used = clinical_info
            
        if text_plot:

            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols, text=list(data.columns.values))
                fig.update_traces(textposition='top center')
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used, text=list(data.columns.values))
                fig.update_traces(textposition='top center')

        else:

            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols)
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used)


    elif len(color_list) > 0:

        if len(x) == len(color_list):
            colors_used = color_list

        if text_plot:

            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols, text=list(data.columns.values))
                fig.update_traces(textposition='top center')
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used, text=list(data.columns.values))
                fig.update_traces(textposition='top center')

        else:
            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols)
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used)

    else:

        if text_plot:

            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols, text=list(data.columns.values))
                fig.update_traces(textposition='top center')
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used, text=list(data.columns.values))
                fig.update_traces(textposition='top center')

        else:

            if len(symbols) > 0:
                fig = px.scatter(x=x, y=y, color=colors_used, symbol = symbols)
                fig.update_traces(textposition='top center')
            
            else:
                fig = px.scatter(x=x, y=y, color=colors_used)

    return fig



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

    df_stat = pro.StatTest(clinical_data, grouped_by, group_1, group_2).apply(data)
    
    # Text the best_n genes with best rate
    best_n = 10

    if df_stat.shape[0] > best_n:

        # Selected the most differencial genes
        p_values_diff = df_stat.loc[df_stat["DiffExpr"] != "Not Diff"]

        p_values_diff = p_values_diff.loc[p_values_diff["FoldChange"] > -ut.np.inf]

        p_values_diff = p_values_diff.loc[p_values_diff["FoldChange"] < ut.np.inf]

        
        p_values_best = p_values_diff.sort_values(by=["P-Value"], ascending=False)

        p_values_best = p_values_best.iloc[:best_n, :]

        # Create a figure to represent Volcano plot
        fig = px.scatter(df_stat, x="FoldChange", y="P-Value", color="DiffExpr")

        for index, row in p_values_best.iterrows():
            fig.add_annotation(
                x=row["FoldChange"], 
                y=row["P-Value"],
                text=row.name,
                showarrow=False,
                yshift=10
            )

        fig.update_traces(textposition='top center')

        return fig

    else:
        
        # Create a figure to represent Volcano plot
        fig = px.scatter(df_stat, x="FoldChange", y="P-Value", color="DiffExpr", text=list(df_stat.index.values))

        fig.update_traces(textposition='top center')

        return fig


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
    
    if isinstance(ini_row, int) and isinstance(fin_row, int):
        # Find duplicated variables in case the user uses the original dataset
        # If the dataset already has the duplicated variables, no actions will be performed
        duplicated_ = (pro.DataDuplicates(axis = 1)).apply(data_duplicated)

        if not duplicated_.empty:

            # Data could be already duplicated or there weren't duplicated but
            # after call duplicated_variables function, the duplicated dataset is obtained
            data_duplicated = ut.copy_object(duplicated_)

            dimens = data_duplicated.shape # [0] -> Rows ; [1] -> Cols
            
            if ini_row > fin_row:
                print("Init row can't be greater than fin row")
                ut.sys.exit(0)
                
                
            if ini_row < 0 or ini_row >= dimens[0]:
                print("Init row exceeds number of rows of DataFrame")
                ut.sys.exit(0)
                
            if fin_row < 0 or fin_row >= dimens[0]:
                print("Fin row exceeds number of rows of DataFrame")
                ut.sys.exit(0)
                

            if (dimens[1] % 2) == 0:
            
                # The duplicate samples are plotted together to assess the correlation.
                n_elems = len(data_duplicated.columns) // 2
                
                n_rows, n_cols = ut.calc_plot_dim(n_elems)
                    
                c_col = 1
                c_row = 1
                
                fig = make_subplots(rows = n_rows, cols = n_cols)
            
            
                for i in range(0, len(data_duplicated.columns), 2):
                        
                    li = (ut.np.log2((data_duplicated.iloc[ini_row:fin_row,i:(i+1)]).mask(data_duplicated.iloc[ini_row:fin_row,i:(i+1)] <= 0.0)).fillna(0.0))
                    li2 = (ut.np.log2((data_duplicated.iloc[ini_row:fin_row,(i+1):(i+2)]).mask(data_duplicated.iloc[ini_row:fin_row,(i+1):(i+2)] <= 0.0)).fillna(0.0))
            
                    if c_col > n_cols:
                        c_col = 1
                        c_row = c_row + 1

                    
                    fig.add_trace(
                        go.Scatter(x = li.values.flatten() , y = li2.values.flatten(), mode="markers", name = li.columns[0]),
                        row = c_row, col = c_col
                    )
                    
                    c_col = c_col + 1
                
                return fig
            
            else:
                print("There is not correspondency between variables")
                ut.sys.exit(0)
                
        else:
            print("There aren't duplicated variables")
    
    else:
        print("Positions must be integer.")
        ut.sys.exit(0)



###############################################################################
        
#                      MACHINE LEARNING VISUALIZATIONS

###############################################################################   


def plot_prec_recall_vs_thresh(testy, predictions):
    """Creates the curve between Precision-Real and Thresholds.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list

    :return: Figure that contains the plot showing the precision recall curve over thresholds.
    :rtype: Figure
    """
  
    precision, recall, thresholds = precision_recall_curve(testy, predictions)
      
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=thresholds, y=precision[:-1], name="Precision", line=dict(dash="dash")))
    
    fig.add_trace(go.Scatter(x=thresholds, y=recall[:-1], name="Recall", line=dict(dash="dash")))
    
    return fig


def plot_roc(testy, predictions):
    """Creates a ROC Curve.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list

    :return: Figure that contains the ROC Curve of the Real Value and Predictions.
    :rtype: Figure
    """
    
    fpr, tpr, _ = roc_curve(testy, predictions)
    
    fig = px.line(
        x = fpr, y = tpr,
    )
    
    fig.add_shape(
        type = "line", line = dict(dash="dash"),
        x0 = 0, x1 = 1, y0 = 0, y1 = 1
    )
    
    return fig


def plot_prc(testy, predictions):
    """Creates a Precision-Recall (PR) Curve.

    :param testy: Real value of a dataset if  a ML algorithm has been applied.
    :type testy: list
    :param predictions: Result of apply ML on the dataset.
    :type predictions: list

    :return: Figure that contains the PR Curve of the Real Value and Predictions.
    :rtype: Figure
    """
    
    precision, recall, _ = precision_recall_curve(testy, predictions)
   
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = recall, y = precision, name = "PR Curve", mode="lines"))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0.5, 0.5], name = "", line = dict(dash="dash")))
    
    return fig


# https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap   
def plot_confusion_matrix(matrix, label_list = None, color = "Viridis"):
    """Create a figure with the Confusion Matrix as Heatmap.

    :param matrix: Contains the elements and data of a confusion matrix.
    :type matrix: list[list]
    :param label_list: Used as value of axis. Defaults to None.
    :type label_list: list
    :param color: color spectrum used on heatmap. Defaults to 'Viridis'.
    :type color: list

    :return: Figure that contains the heatmap showing the colored confusion matrix.
    :rtype: Figure

    """
    
    if isinstance(matrix, ut.np.ndarray):
        
        fig = ff.create_annotated_heatmap(matrix, x = label_list, y = label_list, colorscale = color)
        
        # add colorbar
        fig['data'][0]['showscale'] = True
        
        return fig
    
    else:
        print("Input data must be a matrix")