# -*- coding: utf-8 -*-
"""
@author: Alba Casillas Rodríguez (albacaro@correo.ugr.es)

"""

import math
from src import utils as ut
from abc import abstractmethod



"""
Import Packages and import functions for modeling
"""
from sklearn.linear_model import LogisticRegression as skLogisticRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn.neural_network import MLPClassifier as skMLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,roc_curve, precision_recall_curve, auc

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV


class ModelSelection():
    """This class performs different techniques to the data such as: splitting data,
    over-sampling and under-sampling, normalization, computes confusion matrix and AUC, etc.,
    to the data in order to improve Machine Learning algorithm's performance.
    
    :param data: The data to apply Machine Learning methods. Defaults to None.
        If it is not specified, user must provide the 'source' and 'target' values.
    :type data: DataFrame
    :param source: The features that describe the samples. Defaults to None.
        If it is noy specified, user must provide 'data' value. 
        Otherwise, if it is different to None, 'target' values must be provided.
    :type source: DataFrame
    :param target: The labels to be predicted. Defaults to None.
        If it is noy specified, user must provide 'data' value. 
        Otherwise, if it is different to None, 'source' values must be provided.
    :type target: Series
    :param X_train: All the observations that will be used to train the model.
        Defaults to None. Only can be updated after use 'get_train_test_sample' method.
    :type X_train: DataFrame
    :param Y_train: The dependent variable wich need to be predicted by the model.
        Defaults to None. Only can be updated after use 'get_train_test_sample' method.
    :type Y_train: Series
    :param X_test:  The remaining portion of the independent  variables which 
        will not be used in the training phase and will be used
        to make predictions to test the accuracy of the model.
        Defaults to None. Only can be updated after use 'get_train_test_sample' method.
    :type X_test: DataFrame
    :param Y_test: The labels of the test data. These labels will be used to test 
        the accuracy between actual and predicted categories. Defaults to None.
        Only can be updated after use 'get_train_test_sample' method.
    :type Y_test:  Series
    """
    
    """Constructor method
    """  
    def __init__(self, data = None, source = None, target = None):  
        
        self.data = data 
        self.source = source
        self.target = target
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
    """
    GET
    """
        
    def get_data(self):
        """Returns a DataFrame with the data to apply Machine Learning methods.
        
        :return: The data to apply Machine Learning methods.
        :rtype: DataFrame
        """
        return self.data
    
    def get_source(self):
        """Returns a DataFrame with the attributes used to describe each example.
        
        :return: The features that describe the samples.
        :rtype: DataFrame
        """
        return self.source
    
    def get_target(self):
        """Returns a Series object with the labels to be predicted.
        
        :return: The labels to be predicted.
        :rtype: Series
        """
        return self.target
    
    def get_X_train(self):
        """Returns a DataFrame with all the observations that will be used to
        train the model.
        
        :return: All independent variables used to train the model.
        :rtype: DataFrame
        """
        return self.X_train
    
    def get_Y_train(self):
        """Returns a Series object with the dependent variables which need to be
        predicted by the model.
        
        :return: The dependent variable wich need to be predicted by the model.
        :rtype: Series
        """
        return self.Y_train
    
    def get_X_test(self):
        """Returns a DataFrame with the remaining portion of the independent 
        variables which will not be used in the training phase and will be used
        to make predictions to test the accuracy of the model.
        
        :return: Remaining observations to test the accuracy of the model.
        :rtype: DataFrame
        """
        return self.X_test
    
    def get_Y_test(self):
        """Returns a Series object with the labels of the test data. These labels
        will be used to test the accuracy between actual and predicted categories.
        
        :return: The labels to test the accuracy of the model.
        :rtype: Series
        """
        return self.Y_test
    

    """
    SET
    """
    
    def set_data(self, data):
        """Set the data to apply Machine Learning methods.
        
        :param data: The data to apply Machine Learning methods.
        :type data: DataFrame 
        """
        self.data = data
        
    def set_source(self, source):
        """Set the features that describe the samples.
        
        :param source:  The features that describe the samples.
        :type source: DatFrame 
        """
        self.source = source
        
    def set_target(self, target):
        """Set the labels to be predicted.
        
        :param target: The labels to be predicted.
        :type target: Series 
        """
        self.target = target
        
    def set_X_train(self, x_train):
        """Set the independent variables used to train the model.
        
        :param x_train: All independent variables used to train the model.
        :type x_train: DataFrame 
        """
        self.X_train = x_train
        
    def set_Y_train(self, y_train):
        """Set the dependent variable wich need to be predicted by the model.
        
        :param y_train: The dependent variables wich need to be predicted by the model.
        :type y_train: Series 
        """
        self.Y_train = y_train
        
    def set_X_test(self, x_test):
        """Set the remaining observations to test the accuracy of the model.
        
        :param x_test: Remaining observations to test the accuracy of the model.
        :type x_test: DataFrame 
        """
        self.X_test = x_test
        
    def set_Y_test(self, y_test):
        """Set the labels to test the accuracy of the model.
        
        :param y_test: The labels to test the accuracy of the model.
        :type y_test: str 
        """
        self.Y_test = y_test


    def split_data(self):   
        """Split input data into sources (samples) and target (labels).
        Target data will be the last column of the data.

        """
        x, y = self.get_data().iloc[:,:-1], self.get_data().iloc[:,-1]   
        
        self.set_source(x)
        self.set_target(y)     
    

    def get_train_test_sample(self, test_size = 0.2, resample = False, shuffle = True):
        """Split the data into random train and test subsets.
        If 'resample' is 'True', Smote and Underasmpler techniques will be also applied
        to resample imbalanced data.
        
        :param test_size: If float, should be between 0.0 and 1.0 and represent 
            the proportion of the dataset to include in the test split. 
            If int, represents the absolute number of test samples. 
            If None, the value is set to the complement of the train size. 
            Defaults to '0.2'.
        :type test_size: float, int or None
        :param resample: If 'True', data will be resampled, 'False' will not modify 
            the data. Defaults to 'False'.
        :type resample: bool
        :param shuffle: Whether or not to shuffle the data before splitting.
            Dafaults to 'True'.
        :type shuffle: bool
        """
        
        if resample == True:
            self.over_under_sample(o_samp_str = 0.5, u_samp_str = 0.8)
                                                                                                                                                                                                                                                                                                                                                                                       
        X_train, X_test, Y_train, Y_test = train_test_split(self.get_source(), self.get_target(), test_size = test_size, shuffle = shuffle)
        
        self.set_X_train(X_train)
        self.set_X_test(X_test)
        self.set_Y_train(Y_train)
        self.set_Y_test(Y_test)



    def apply_smote(self, sampling_strategy = "auto", random_state = None, k_neighbors = 5, n_jobs = None):
        """Applies SMOTE (Synthetic Minority Over-sampling Technique) to perform
        over-sampling.
        
        For more information, see imbalanced learn documentation:
        https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
        
        :param sampling_strategy: Sampling information to resample the data set.
            Defaults to 'auto'.
        :type sampling_strategy: float, str, dict or callable
        :param random_state: Control the randomization of the algorithm. Defaults to None.
        :type random_state: int, RandomState instance or None
        :param k_neighbors: Number of neighbours to used to construct synthethic samples.
            Defaults to '5'.
        :type k_neighbors: int or Object
        :param n_jobs: Number of CPU cores used during the cross-validation loop. Defaults to None.
        :type n_jobs: int
        """
        
        over = SMOTE(sampling_strategy = sampling_strategy, random_state = random_state, k_neighbors = k_neighbors, n_jobs = n_jobs)
        
        X_res, Y_res = over.fit_resample(self.get_source(), self.get_target())
        self.set_source(X_res)
        self.set_target(Y_res)  



    def apply_undersampler(self, sampling_strategy = "auto", random_state = None, replacement = False):
        """Applies RandomUnderSampler  to perform random under-sampling.
        The method under-sample the majority class(es) by randomly picking samples 
        with or without replacement.
        
        For more information, see imbalanced learn documentation:
        https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
        
        :param sampling_strategy: Sampling information to resample the data set.
            Defaults to 'auto'.
        :type sampling_strategy: float, str, dict or callable
        :param random_state: Control the randomization of the algorithm. Defaults to None.
        :type random_state: int, RandomState instance or None
        :param replacement: Whether the sample is with or without replacement. Defaults to False.
        :type replacement: bool
        """
        
        under = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state = random_state, replacement = replacement)

        X_res, Y_res = under.fit_resample(self.get_source(), self.get_target())
        self.set_source(X_res)
        self.set_target(Y_res) 
    
    
    def over_under_sample(self, o_samp_str = "auto", u_samp_str = "auto", o_random_st = None, u_random_st = None):
        """Combines over and under sample to avoid over-fitting or missing too much information.
        Both techniques will be combines by using Pipeline from imblearn, that provides a pipeline
        by applying a list of transformations, and resamples, with a final estimator.
        
        :param o_samp_str: For SMOTE, sampling information to resample the data set.
            Defaults to 'auto'.
        :type o_samp_str: float, str, dict or callable
        :param u_samp_str: For UnderSampler, sampling information to resample the data set.
            Defaults to 'auto'.
        :type u_samp_str: float, str, dict or callable
        :param o_random_st: For SMOTE, control the randomization of the algorithm. 
            Defaults to None.
        :type o_random_st: int, RandomState instance or None
        :param u_random_st: For UnderSampler, control the randomization of the algorithm. 
            Defaults to None.
        :type u_random_st: int, RandomState instance or None
        """
        
        over = SMOTE(sampling_strategy = o_samp_str)
        
        under = RandomUnderSampler(sampling_strategy = u_samp_str)
        
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps)
        
        X_res, Y_res = pipeline.fit_resample(self.get_source(), self.get_target()) 
        
        self.set_source(X_res)
        self.set_target(Y_res) 



    def standarize(self):
        """Normalize the data standarizing features by removing the man ands scaling to unit variance.
        With it, the mean will be 0 and the standard deviation will be 1, following the equation:
        X_stand = (x - mean(x)) / standard deviation(x)
        
        For more information, see:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html     
        """
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        scaler = StandardScaler()
        
        # Fit only to the training data
        scaler.fit(self.get_X_train())
        
        # Now apply the transformations to the data:
        X_train = scaler.transform(self.get_X_train())
        X_test = scaler.transform(self.get_X_test())     
           
        self.set_X_train(X_train)
        self.set_X_test(X_test)
    
 
    def get_predictions(self, probabilities, threshold=0.5):    
        """Get prediction values.
        
        :param probabilities: list of probabilities of the X data that represent a percentage of its result
        :type probabilities: list 
        :param threshold: a limit to indicates when the classification will be 0 or 1
        :type threshold: float

        :return: a predicted values by probabilities
        :rtype: list
        """

        predict_res = ut.np.where(probabilities <= threshold, 0, 1)
     
        return predict_res 
        
    def predict_proba(self, model):
        """Predcit probabilities for a given model
        
        :param model: Model to predict a given value
        :type model: object
        :return: predicted values
        :rtype: array
        """

        return model.get_model().predict_proba(self.get_X_test())[:,1]
 

    def confusion_matrix(self, Y_pred, labels = None, sample_weight = None):
        """Compute confusion matrix to evaluate the accuracy of a classification.
        By definition, a confusion matrix C is such that Cij is equal to the number
        of observations known to be in group i and predicted to be in group j.
        
        For more information, see scikit learn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            
        :param Y_pred: Estimated targets as returned by a classifier.
        :type Y_pred: array of shape n_samples
        :param labels: List of labels to index the matrix. Defaults to None.
        :type labels: array of shape n_classes or None
        :param sample_weight: Sample weights. Defaults to None.
        :type sample_weight: array of shape n_samples or None

        :return: Confusion matrix whose i-th row and j-th column entry indicates 
            the number of samples with true label being i-th class and 
            predicted label being j-th class.
        :rtype: ndarray
        """
        
        return(confusion_matrix(self.get_Y_test(), Y_pred, sample_weight = sample_weight))
    
           
    def calc_auc(self, predictions, curve = "roc"):
        """AUC stands for "Area under the ROC Curve". That is, AUC measures the entire
        two-dimensional area underneath the entire ROC curve from (0,0) to (1,1)
        The Precission-Recall AUC is just like the ROC AUC. in that it summarizes the curve with a range
        of threshold values as a single score.
        
        :param predictions: Target scores, can either be probability estimates 
            of the positive class, or non-thresholded measure of decisions.
        :type predictions: ndarray of shape n_samples
        :param curve: If 'roc', computes Receiver Operatic Characteristic (ROC) curve,
            if 'precision-recall', computes Precision-Recall curve.
        :type curve: str

        :return: Returns the Area Under the Curve (AUC)
        :rtype: float
        """

        # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/#:~:text=The%20Precision%2DRecall%20AUC%20is,a%20model%20with%20perfect%20skill.
        auc_score = -1
        
        if curve == "roc":
        
            fpr, tpr, _ = roc_curve(self.get_Y_test(), predictions)
            auc_score = auc(fpr, tpr)  
            
        elif curve == "precision-recall":
            
            precision, recall, _ = precision_recall_curve(self.get_Y_test(), predictions)
            auc_score = auc(recall, precision)    
            
        else:
            print("Invalid state of curve passed.")
        
        return auc_score 


    def cross_validation(self, model, n_splits = 10, n_jobs = None):
        """Computes K-fold Corss-Validation and evaluate the metric(s) by using
        StratifiedKFold and cross_validate from scikit-learn
        
        StratifiedKFold:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        cross_validate:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html   
        
        :param model: Machine Learning algorithm to apply cross-validation.
        :type model: 
        :param n_splits: Number of folds. Must be at least 2. Defaults to 10.
        :type n_splits: int
        :param n_jobs: Number of jobs to run in parallel. Training the estimator
            and computing the score are parallelized over the cross-validation splits.
            Defaults to None
        :type n_jobs: int

        :return: The function will return a dictionary containing the metrics "accuracy",
            "precision", "recall", and "f1" for boith training and validation sets.
        :rtype: dict
        """

        # Indices obtained according to the number of partitions for the cross validation
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        cv = StratifiedKFold(n_splits = n_splits , shuffle = True, random_state = 1)
        
        scores = ['accuracy', 'precision', 'recall', 'f1']
        
        results = cross_validate(estimator = model,
                                 X = self.get_source(),
                                 y = self.get_target(),
                                 cv = cv,
                                 scoring = scores,
                                 return_train_score = True)
        
        return {"Training Accuracy scores": results['train_accuracy'],
                "Mean Training Accuracy": results['train_accuracy'].mean()*100,
                
                "Training Precision scores": results['train_precision'],
                "Mean Training Precision": results['train_precision'].mean(),
                
                "Training Recall scores": results['train_recall'],
                "Mean Training Recall": results['train_recall'].mean(),
                
                "Training F1 scores": results['train_f1'],
                "Mean Training F1 Score": results['train_f1'].mean(),
                
                "Validation Accuracy scores": results['test_accuracy'],
                "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
                
                "Validation Precision scores": results['test_precision'],
                "Mean Validation Precision": results['test_precision'].mean(),
                
                "Validation Recall scores": results['test_recall'],
                "Mean Validation Recall": results['test_recall'].mean(),
                
                "Validation F1 scores": results['test_f1'],
                "Mean Validation F1 Score": results['test_f1'].mean()
                }
    

class Model():
    """This is a conceptual class representation of a model that can be used
    with Machine Learning algorithms.
    
    :param name: Name of the model.
    :type name: str
    :param model: Model to work with.
    :type model: object
    """
        
    def __init__(self, name, model = None):  
        self.name = name
        self.model = model
        
    def get_name(self):
        """Returns a string with the name of the model.
        
        :return: The name of the model.
        :rtype: str
        """
        return self.name

    def get_model(self):
        """Returns the model to work with.
        
        :return: The model to work with.
        :rtype: object
        """
        return self.model

        
    def set_name(self, name):
        """Set the name of the model.
        
        :param name: The name of the model.
        :type name: str 
        """
        self.name = name
        
    def set_model(self, model):
        """Set the model to work with.
        
        :param model: The model to work with.
        :type model: object
        """
        self.model = model

    def fit(self, X_train, Y_train):
        """Fit (train) the model.
       
        :param X_train: All independent variables used to train the model.
        :type X_train: DataFrame
        :param Y_train: The dependent variables wich need to be predicted by the model.
        :type Y_train: Series
        """
        
        (self.get_model()).fit(X_train, Y_train)
              
    def predict(self, X_test):
        """ Given an unlabeled observations X, returns the predicted labels Y.
       
        :param X_test: Remaining observations to test the accuracy of the model.
        :type X_test: DataFrame
        :return: Returns the predicted labels Y.
        :rtype: Series
        """
        return (self.get_model()).predict(X_test.values)
    
    @abstractmethod 
    def get_metrics(self, Y_true, Y_pred, y_pred_proba=None):    
        raise NotImplementedError("Must override get_metrics")
    
            
class ScikitLearnModel(Model):
    """This is a conceptual class representation of a model that is provided by
    the Scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param model: Model to work with.
    :type model: object
    """
    
    def __init__(self, name, model = None):
        # Call the __init__ function of the father class
        super().__init__(name, model)

        
    @abstractmethod 
    def hyperparameter_tuning(X_train, Y_train):
        raise NotImplementedError("Must override hyperparameter_tuning")
    

    def get_metrics(self, Y_true, Y_pred, y_pred_proba=None): 
        """Computes the metrics "accuracy", "precision", "recall", "specifity" and "f1" 
        for a specified model and returns a Dataframe with all the information.
       
        :param Y_true: Ground truth (correct) target values.
        :type Y_true: array
        :param Y_pred: Estimated targets as returned by a classifier.
        :type Y_pred: array
        :return: Returns a DataFrame with all the metrics.
        :rtype: DataFrame
        """
        
        cv_scores = []  

        cm = confusion_matrix(Y_true,Y_pred)

        # Precision indicates the proportion of positive indentifications that are actually correct
        precision = precision_score(Y_true, Y_pred)
        
        # Recall (Sensivity) indicates the proportion of actual positives that were identified correctly
        recall = recall_score(Y_true, Y_pred)
        
        # Specificity indicates the proportion of actual negatives, which got predicted as the negative
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])  
        
        # F1-Score is the weighted average of Precision and Recall, taking both false positives and
        # false negatives into account
        f1 = f1_score(Y_true, Y_pred)  
        
        # Accuracy ratio of correctly  predicted observation to the total observations
        accuracy = accuracy_score(Y_true, Y_pred)
    
        cv_scores.append([self.get_name(), precision, recall, specificity, f1, accuracy])    
        results_df = ut.pd.DataFrame(cv_scores, columns=['model_name', 'precision', 'recall', 'specificity', 'f1', 'accuracy'])
    
        return results_df
     
        
class LogisticRegression(ScikitLearnModel):
    """This is a conceptual class that represents Logistic Regression algorithm
    from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param penalty: Specify the norm of the penalty, where 'none' means no penalty is added,
        'l2' uses Ridge regularization, 'l1' uses Lasso regularization, and 'elasticnet'
        means that both 'l1' and 'l2' penalty terms are added. Defaults to 'l2'.
    :type penalty: 'l1','l2','elasticnet', 'none'
    :param dual: Dual or primal formulation. Dual formulation is only implemented 
        for l2 penalty with liblinear solver. Defauls to 'False'.
    :type dual: bool
    :param tol: Tolerance for stopping criteria. Defaults to '1e-4'.
    :type tol: float
    :param C: Inverse of regularization strenght, must be a positive float. 
        Smaller values specify stronger regularization. Defaults to '1.0'.
    :type C: float
    :param fit_intercept: Specifies if a constant (bias or intercept) should be added
        to the decision function. Defaults to 'True'.
    :type fit_intercept: bool
    :param intercept_scaling: Useful only when the solver 'liblinear' is used and
        'fit_intercept' True. In this case, x becomes [x, self.intercept_scaling].
        Defaults to '1'.
    :type intercept_scaling: float
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, 'balanced', or None
    :param random_state: Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None
    :param solver: Algorith to use in the optimization problem. Defaults to 'lbfgs'.
    :type solver: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
    :param max_iter: Maximum number of iterations taken for the solvers to converge.
        Defaults to '100'.
    :type max_iter: int
    :param multi_class: If the option chosen is 'ovr', then a binary problem is fit 
        for each label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, even when the data is binary.
        Defaults to 'auto'.
    :type multi_class: {'auto' , 'ovr' , 'multinomial'}
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity. Defaults to '0'.
    :type verbose: int
    :param warm_start: If 'True', reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to 'False'.
    :type warm_start: bool
    :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class='ovr'.
        Defaults to None.
    :type n_jobs: int
    :param l1_ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
        Only used if penalty = 'elasticnetV. Defaults to None.
    :type l1_ratio: float
    """
    
    
    def __init__(self, name, penalty = 'l2', dual = False, tol = math.pow(10,-4),
                 C = 1.0, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None,
                 solver = 'lbfgs', max_iter = 100, multi_class = 'auto', verbose = 0, warm_start = False, 
                 n_jobs = None, l1_ratio = None):
        
        super().__init__(name)
        
        self.set_model(skLogisticRegression(penalty = penalty, dual = dual, tol = tol,
                 C = C, fit_intercept = fit_intercept, intercept_scaling = intercept_scaling, class_weight = class_weight, random_state = random_state,
                 solver = solver, max_iter = max_iter, multi_class = multi_class, verbose = verbose, warm_start = warm_start, 
                 n_jobs = n_jobs, l1_ratio = l1_ratio))
        
        
    def hyperparameter_tuning(self, X_train, Y_train, penalty = None, dual = None, tol = None, C = None, fit_intercept = None, 
                              intercept_scaling = None, class_weight = None, random_state = None,
                              solver = None, max_iter = None, multi_class = None, verbose = None, warm_start = None,
                              n_jobs = None, l1_ratio = None):
        """Computes a hyperparameter tuning for Logist Regression. 
            User can provides a list of values for a specified parameter to perform
            the GridSearchCV method from scikit-learn.
    
        :param X_train: All independent variables used to train the model.
        :type name: DataFrame
        :param Y_train: The dependent variables wich need to be predicted by the model.
        :type Y_train: Series
        :param penalty: Specify the norm of the penalty, where 'none' means no penalty is added,
            'l2' uses Ridge regularization, 'l1' uses Lasso regularization, and 'elasticnet'
            means that both 'l1' and 'l2' penalty terms are added. Defaults to None.
        :type penalty: list, None
        :param dual: Dual or primal formulation. Dual formulation is only implemented 
            for l2 penalty with liblinear solver. Defauls to None.
        :type dual: list, None
        :param tol: Tolerance for stopping criteria. Defaults to None.
        :type tol: list, None
        :param C: Inverse of regularization strenght, must be a positive float. 
            Smaller values specify stronger regularization. Defaults to None.
        :type C: flist, None
        :param fit_intercept: Specifies if a constant (bias or intercept) should be added
            to the decision function. Defaults to None.
        :type fit_intercept: list, None
        :param intercept_scaling: Useful only when the solver 'liblinear' is used and
            'fit_intercept' True. In this case, x becomes [x, self.intercept_scaling].
            Defaults to None.
        :type intercept_scaling: list, None
        :param class_weight: Weights associated with classes in the form {class_label: weight}.
            Defaults to None.
        :type class_weight: list, None
        :param random_state: Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
            Defaults to None.
        :type random_state: list, None
        :param solver: Algorith to use in the optimization problem. Defaults to None.
        :type solver: list, None
        :param max_iter: Maximum number of iterations taken for the solvers to converge.
            Defaults to None.
        :type max_iter: list, None
        :param multi_class: If the option chosen is 'ovr', then a binary problem is fit 
            for each label. For 'multinomial' the loss minimised is the multinomial loss fit
            across the entire probability distribution, even when the data is binary.
            Defaults to None.
        :type multi_class: list, None
        :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.
        :type verbose: list, None
        :param warm_start: If 'True', reuse the solution of the previous call to fit
            as initialization, otherwise, just erase the previous solution. Defaults to None.
        :type warm_start: list, None
        :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class='ovr'.
            Defaults to None.
        :type n_jobs: list, None
        :param l1_ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
            Only used if penalty = 'elasticnetV. Defaults to None.
        :type l1_ratio: list, None
        """

        param_grid = {}
        model_lr = skLogisticRegression()
        
        # https://stackoverflow.com/questions/2912615/how-to-iterate-over-function-arguments
        arguments = locals()
        del arguments['X_train']
        del arguments['Y_train']
        
        for arg in arguments.items():
            
            if isinstance(arg[1], list):
                param_grid[arg[0]] = arg[1]
        
        clf = GridSearchCV(model_lr, param_grid)
        clf.fit(X_train, Y_train)
        
        best_params = clf.best_params_
        
        # https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
        self.set_model(skLogisticRegression(**best_params))



class RandomForest(ScikitLearnModel):
    """This is a conceptual class that represents Random Forest classifier algorithm
    from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param n_estimators: Number of trees in the forest. Defaults to '100'.
    :type n_estimators: int
    :param criterion: Function to measure the quality of a split. Defaults to 'gini'.
    :type criterion: {'gini', 'entropy', 'log_loss'}.
    :param max_depth: The maximum depth of the tree. If None, nodes are expanded
        until all leaves are pure or contains less than 'min_sample_split' samples.
        Defaults to None.
    :type max_depth: int, None
    :param min_samples_split: The minimum number of samples required to split an
        internal node. Defaults to '2'.
    :type min_samples_split: int or float
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        Defaults to '1'.
    :type min_samples_leaf: int or float
    :param min_weight_fraction_leaf: The minimum weighted fraction of the sum total
        of weights (of all the input samples) required to be at a leaf node.
        Defaults to '0.0'.
    :type min_weight_fraction_leaf: float
    :param max_features: The number of features to consider when looking for
        the best split. Defaults to 'sqrt'.
    :type max_features:{'sqrt', 'log2', None}, int or float
    :param max_leaf_nodes: Grow trees with 'max_leaf_nodes' in best-first fashion.
        Defaults to None.
    :type max_leaf_nodes: int, None
    :param min_impurity_decrease: A node will be split if this split induces a decrease
        of the impurity greater than or equal to this value. Defaults to '0.0'.
    :type min_impurity_decrease: float
    :param bootstrap: Whether bootstrap samples are used when building trees. 
        If 'False', the whole dataset is used to build each tree. Defauls to 'True'.
    :type bootstrap: bool
    :param oob_score: Wheter to use out-of-bag samples to estimate the generalization score.
        Only available if boostrap is 'True'. Defaults to 'False'.
    :type oob_score: bool
    :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class='ovr'.
        Defaults to None.
    :type n_jobs: int
    :param random_state: Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity. Defaults to '0'.
    :type verbose: int
    :param warm_start: If 'True', reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to 'False'.
    :type warm_start: bool
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, 'balanced', 'balanced_subsample', or None
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
        Defaults to '0.0'.
    :type ccp_alpha: non-negative float
    :param max_samples: If boostrap is 'True', the number of samples to draw from X to 
        train each base estimator. Defaults to None.
    :type max_samples: int, float, None
    """
    
    def __init__(self, name, n_estimators = 100, criterion = 'gini', max_depth = None, min_samples_split = 2,
               min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = 'sqrt', max_leaf_nodes = None, 
               min_impurity_decrease = 0.0, bootstrap = True, oob_score = False, n_jobs = None, random_state = None, verbose = 0,
               warm_start = False, class_weight = None, ccp_alpha = 0.0, max_samples = None):
        
        super().__init__(name)

        self.set_model(skRandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
               min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, 
               min_impurity_decrease = min_impurity_decrease, bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state, verbose = verbose,
               warm_start = warm_start, class_weight = class_weight, ccp_alpha = ccp_alpha, max_samples = max_samples))
        
        
    def hyperparameter_tuning(self, X_train, Y_train, n_estimators = None, criterion = None, max_depth = None, min_samples_split = None,
               min_samples_leaf = None, min_weight_fraction_leaf = None, max_features = None, max_leaf_nodes = None, 
               min_impurity_decrease = None, bootstrap = None, oob_score = None, n_jobs = None, random_state = None, verbose = None,
               warm_start = None, class_weight = None, ccp_alpha = None, max_samples = None):
        """Computes a hyperparameter tuning for Random Forest Classifier. 
            User can provides a list of values for a specified parameter to perform
            the GridSearchCV method from scikit-learn.
    
        :param X_train: All independent variables used to train the model.
        :type name: DataFrame
        :param Y_train: The dependent variables wich need to be predicted by the model.
        :type Y_train: Series
        :param n_estimators: Number of trees in the forest. Defaults to None.
        :type n_estimators:  list, None
        :param criterion: Function to measure the quality of a split.  Defaults to None.
        :type criterion:  list, None
        :param max_depth: The maximum depth of the tree. If None, nodes are expanded
            until all leaves are pure or contains less than 'min_sample_split' samples.
            Defaults to None.
        :type max_depth:  list, None
        :param min_samples_split: The minimum number of samples required to split an
            internal node.  Defaults to None.
        :type min_samples_split:  list, None
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
             Defaults to None.
        :type min_samples_leaf:  list, None
        :param min_weight_fraction_leaf: The minimum weighted fraction of the sum total
            of weights (of all the input samples) required to be at a leaf node.
             Defaults to None.
        :type min_weight_fraction_leaf:  list, None
        :param max_features: The number of features to consider when looking for
            the best split. Defaults to None..
        :type max_features: list, None
        :param max_leaf_nodes: Grow trees with 'max_leaf_nodes' in best-first fashion.
             Defaults to None.
        :type max_leaf_nodes: list, None
        :param min_impurity_decrease: A node will be split if this split induces a decrease
            of the impurity greater than or equal to this value. Defaults to None.
        :type min_impurity_decrease: list, None
        :param bootstrap: Whether bootstrap samples are used when building trees. 
            If 'False', the whole dataset is used to build each tree.  Defaults to None.
        :type bootstrap: list, None
        :param oob_score: Wheter to use out-of-bag samples to estimate the generalization score.
            Only available if boostrap is 'True'. Defaults to None.
        :type oob_score: list, None
        :param n_jobs: Number of CPU cores used when parallelizing over classes of multi_class='ovr'.
            Defaults to None.
        :type n_jobs: list, None
        :param random_state: Controls both the randomness of the bootstrapping of 
            the samples used when building trees and the sampling of the features 
            to consider when looking for the best split at each node.
            Defaults to None.
        :type random_state: list, None
        :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.  Defaults to None.
        :type verbose: list, None
        :param warm_start: If 'True', reuse the solution of the previous call to fit
            as initialization, otherwise, just erase the previous solution.  Defaults to None.
        :type warm_start: list, None
        :param class_weight: Weights associated with classes in the form {class_label: weight}.
            Defaults to None.
        :type class_weight: list, None
        :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
             Defaults to None.
        :type ccp_alpha: list, None
        :param max_samples: If boostrap is 'True', the number of samples to draw from X to 
            train each base estimator. Defaults to None.
        :type max_samples: list, None
        """
        
        param_grid = {}
        model_rf = skRandomForestClassifier()
        
        # https://stackoverflow.com/questions/2912615/how-to-iterate-over-function-arguments
        arguments = locals()
        del arguments['X_train']
        del arguments['Y_train']
        
        for arg in arguments.items():
            
            if isinstance(arg[1], list):
                param_grid[arg[0]] = arg[1]
        
        clf = GridSearchCV(model_rf, param_grid)
        clf.fit(X_train, Y_train)
        
        best_params = clf.best_params_
        
        # https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
        self.set_model(skRandomForestClassifier(**best_params))



class SupportVectorClassif(ScikitLearnModel):
    """This is a conceptual class that represents Support Vector Machine algorithm
    from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param C: Regularization parameter. The strenght of the regularization is 
        inversely proportional to C. Defaults to '1.0'.
    :type C: float
    :param kernel: Specifies the kernel type to be used in the algorithm. 
        Defaults to 'rbf'
    :type kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable
    :param degree: Degree of the polynomial kernel function when kernel is 'poly'.
        Defaults to '3'.
    :type degree: int
    :param gamma: Kernek coefficient fot 'rbf', 'poly' and 'sigmoid'.
        Defaults to 'scale'.
    :type gamma: {'scale', 'auto'} or float
    :param coef0: Independent term in kernel function, when kernel is 'poly' or
        'sigmoid'. Defaults to '0.0'.
    :type coef0: float
    :param shrinking: Whether to use the shrinking heuristic. Defaults to 'True'.
    :type shrinking: bool
    :param probability: Wheter to enable probability estimates. Deafuls to 'False'.
    :type probability: bool
    :param tol: Tolerance for stopping criteria. Defaults to '1e-3'.
    :type tol: float
    :param cache_size: Specify the size of the kernelk cache (in MB). Defaults to '200'.
    :type cache_size: float
    :param class_weight: Weights associated with classes in the form {class_label: weight}.
        Defaults to None.
    :type class_weight: dict, 'balanced', or None
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.  Defaults to None.
    :type verbose: bool
    :param max_iter: Hard limit on iterations within solver, or -1 for no limit.
        Defaults to -1.
    :type max_iter: int
    :param decision_function_shape: Wheter to return a one-vs-rest ('ovr') or 
        one-vs-one ('ovo') decision function. Defaults to 'ovr'.
    :type decision_function_shape: {'ovo', 'ovr'}
    :param break_ties: If 'True', 'decision_functoin_shape' = 'ovr' and number of
        classes > 2, predict will break ties according to the confidence values
        of 'decision_function'. Defaults to 'False'.
    :type break_ties: bool
    :param random_state: Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
        Defaults to None.
    :type random_state: int, RandomState instance, or None
    """
    
    def __init__(self, name, C = 1.0, kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, 
               shrinking = True, probability = False, tol = math.pow(10,-3), cache_size = 200, class_weight = None, 
               verbose = False, max_iter = -1, decision_function_shape = 'ovr', break_ties = False, random_state = None):
        
        super().__init__(name)

        self.set_model(SVC(C = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, 
               shrinking = shrinking, probability = probability, tol = tol, cache_size = cache_size, class_weight = class_weight, 
               verbose = verbose, max_iter = max_iter, decision_function_shape = decision_function_shape, break_ties = break_ties, random_state = random_state))
        

    def hyperparameter_tuning(self, X_train, Y_train, C = None, kernel = None, degree = None, gamma = None, coef0 = None, shrinking = None,
                             probability = None, tol = None, cache_size = None, class_weight = None,
                             verbose = None, max_iter = None, decision_function_shape = None, break_ties = None, random_state = None):
        """Computes a hyperparameter tuning for Random Forest Classifier. 
            User can provides a list of values for a specified parameter to perform
            the GridSearchCV method from scikit-learn.
    
        :param X_train: All independent variables used to train the model.
        :type name: DataFrame
        :param Y_train: The dependent variables wich need to be predicted by the model.
        :type Y_train: Series
        :param C: Regularization parameter. The strenght of the regularization is 
            inversely proportional to C.  Defaults to None.
        :type C: list, None
        :param kernel: Specifies the kernel type to be used in the algorithm. 
            Defaults to None.
        :type kernel: list, None
        :param degree: Degree of the polynomial kernel function when kernel is 'poly'.
             Defaults to None.
        :type degree: list, None
        :param gamma: Kernek coefficient fot 'rbf', 'poly' and 'sigmoid'.
             Defaults to None.
        :type gamma: list, None
        :param coef0: Independent term in kernel function, when kernel is 'poly' or
            'sigmoid'. Defaults to None.
        :type coef0: list, None
        :param shrinking: Whether to use the shrinking heuristic. Defaults to None.
        :type shrinking: list, None
        :param probability: Wheter to enable probability estimates.  Defaults to None.
        :type probability: list, None
        :param tol: Tolerance for stopping criteria.  Defaults to None.
        :type tol: list, None
        :param cache_size: Specify the size of the kernelk cache (in MB).  Defaults to None.
        :type cache_size: list, None
        :param class_weight: Weights associated with classes in the form {class_label: weight}.
            Defaults to None.
        :type class_weight: list, None
        :param verbose:  For the liblinear and lbfgs solvers set verbose to any positive
                number for verbosity.  Defaults to None.
        :type verbose: list, None
        :param max_iter: Hard limit on iterations within solver, or -1 for no limit.
             Defaults to None.
        :type max_iter: list, None
        :param decision_function_shape: Wheter to return a one-vs-rest ('ovr') or 
            one-vs-one ('ovo') decision function. Defaults to None.
        :type decision_function_shape: list, None
        :param break_ties: If 'True', 'decision_functoin_shape' = 'ovr' and number of
            classes > 2, predict will break ties according to the confidence values
            of 'decision_function'.  Defaults to None.
        :type break_ties: list, None
        :param random_state: Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
            Defaults to None.
        :type random_state: list, None
        """
        
        param_grid = {}
        model_svc = SVC()
        
        # https://stackoverflow.com/questions/2912615/how-to-iterate-over-function-arguments
        arguments = locals()
        del arguments['X_train']
        del arguments['Y_train']
        
        for arg in arguments.items():
            
            if isinstance(arg[1], list):
                param_grid[arg[0]] = arg[1]
        
        clf = GridSearchCV(model_svc, param_grid)
        clf.fit(X_train, Y_train)
        
        best_params = clf.best_params_
        
        # https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
        self.set_model(SVC(**best_params))



class NeuralNetwork(ScikitLearnModel):
    """This is a conceptual class that represents Neural Network algorithm
    from scikit-learn library.

    :param name: Name of the model.
    :type name: str
    :param hidden_layer_sizes: The ith element represents the number of neurons in the
        ith hidden layer. Defaults to '(100,)'.
    :type hidden_layer_sizes: tuple, length = n_layers - 2
    :param activation: Activation function for the hidden layer. Defaults to 'relu'.
    :type activation: {'identity', 'logistic', 'tanh', 'relu'}
    :param solver: The solver for weight optimization. Defaults to 'adam'.
    :type solver: {'lbfgs', 'sgd', 'adam'}
    :param alpha: Strength of the L2 regularization term (which is divided by the sample
        size when added to the loss). Defaults to '0.0001'.
    :type alpha: float
    :param batch_size: Size of minibatches for stochastic optimizers. Defaults to 'auto'.
    :type batch_size: int, str
    :param learning_rate: Learning rate schedule for weight purposes. Defaults to 'constant'.
    :type learning_rate: {'constant', 'invscaling', 'adaptive'}
    :param learning_rate_init: The initial learning rate used. It controls the step-size
        in updating the weights. Only when solver is 'sgd' or 'adam'. Defaults to '0.001'.
    :type learning_rate_init: float
    :param power_t: The exponent for inverse scaling learning rate. It is used in updating
        effective learning rate when the 'learning_rate' is 'invscaling'. Only when solver 
        is 'sgd'. Defaults to '0.5'
    :type power_t: float
    :param max_iter: Maximum number of iterations. Defaults to '200'.
    :type max_iter: int
    :param shuffle: Whether to shuffle samples in each iteration. Only when solver is 'sgd' 
        or 'adam'. Defaults to 'True'.
    :type shuffle: bool
    :param random_state: Determines random number generation for weights and bias initialization,
        train-test split and batch sampling. Defaults to None.
    :type random_state: int, RandomInstance, None
    :param tol:Tolerance for stopping criteria. Defaults to '1e-4'.
    :type tol: float
    :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.  Defaults to 'False'.
    :type verbose: bool
    :param warm_start: If 'True', reuse the solution of the previous call to fit
        as initialization, otherwise, just erase the previous solution. Defauls to 'False'.
    :type warm_start: bool
    :param momentum: Momentum for gradient descent update. Should be between 0 and 1. 
        Defaults to '0.9'.
    :type momentum: float
    :param nesterovs_momentum: Wheter to use Nesterov's momentum. Only when solves is 'sgd'
        and momentum > 0. Defaults to 'True'.
    :type nesterovs_momentum: bool
    :param early_stopping: Whether to use early stopping to terminate training when 
        validation score is not improving. Defaults to 'False'.
    :type early_stopping: bool
    :param validation_fraction: The proportion of training data to set aside as validation 
        set fot early stopping. Must be between 0 and 1. Only if early stopping is 'True'.
        Defaults to '0.1'.
    :type validation_fraction: float
    :param beta_1: Exponential decay rate for estimates of first moment vector in adam, 
        should be between 0 and 1. Only used when solver is 'adam'. Defaults to '0.9'.
    :type beta_1: float
    :param beta_2: Exponential decay rate for estimates of second moment vector in adam, 
        should be between 0 and 1. Only used when solver is 'adam'. Defaults to '0.999'.
    :type beta_2: float
    :param epsilon: Value for numerical stability in adam. Only used when solver is 'adam'. 
        Defaults to '1e-8'.
    :type epsilon: float
    :param n_iter_no_change: Maximum number of epoch to not meet tol improvement. 
        Only when solver is 'sgd' or 'adam'. Defaults to '10'.
    :type n_iter_no_change: int
    :param max_fun: Maximum number of loss functions calls. Only when solver is 'lbfgs'. 
        Defaults to '15000'.
    :type max_fun: int
    """
    
    def __init__(self, name, hidden_layer_sizes = (100,), activation = 'relu', solver = 'adam',
               alpha = 0.0001, batch_size = 'auto', learning_rate = 'constant', learning_rate_init = 0.001, power_t = 0.5,
               max_iter = 200, shuffle = True, random_state = None, tol = math.pow(10,-4), verbose = False, warm_start = False,
               momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, 
               beta_2 = 0.999, epsilon =  math.pow(10,-8), n_iter_no_change = math.pow(10,-8), max_fun = 15000):
        
        super().__init__(name)

        self.set_model(skMLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver,
               alpha = alpha, batch_size = batch_size, learning_rate = learning_rate, learning_rate_init = learning_rate_init, power_t = power_t,
               max_iter = max_iter, shuffle = shuffle, random_state = random_state, tol = tol, verbose = verbose, warm_start = warm_start,
               momentum = momentum, nesterovs_momentum = nesterovs_momentum, early_stopping = early_stopping, validation_fraction = validation_fraction, beta_1 = beta_1, 
               beta_2 = beta_2, epsilon = epsilon, n_iter_no_change = n_iter_no_change, max_fun = max_fun))
        

    def hyperparameter_tuning(self, X_train, Y_train, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, 
                                   learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, 
                                   warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction,
                                   beta_1, beta_2, epsilon, n_iter_no_change, max_fun):
        
        """Computes a hyperparameter tuning for Random Forest Classifier. 
            User can provides a list of values for a specified parameter to perform
            the GridSearchCV method from scikit-learn.
    
        :param X_train: All independent variables used to train the model.
        :type name: DataFrame
        :param Y_train: The dependent variables wich need to be predicted by the model.
        :type Y_train: Series
        :param hidden_layer_sizes: The ith element represents the number of neurons in the
            ith hidden layer. Defaults to None.
        :type hidden_layer_sizes:  list, None
        :param activation: Activation function for the hidden layer. Defaults to None.
        :type activation:  list, None
        :param solver: The solver for weight optimization. Defaults to None.
        :type solver:  list, None
        :param alpha: Strength of the L2 regularization term (which is divided by the sample
            size when added to the loss). Defaults to None.
        :type alpha:  list, None
        :param batch_size: Size of minibatches for stochastic optimizers. Defaults to None.
        :type batch_size:  list, None
        :param learning_rate: Learning rate schedule for weight purposes. Defaults to None.
        :type learning_rate: list, None
        :param learning_rate_init: The initial learning rate used. It controls the step-size
            in updating the weights. Only when solver is 'sgd' or 'adam'. Defaults to None.
        :type learning_rate_init:  list, None
        :param power_t: The exponent for inverse scaling learning rate. It is used in updating
            effective learning rate when the 'learning_rate' is 'invscaling'. Only when solver 
            is 'sgd'. Defaults to None.
        :type power_t:  list, None
        :param max_iter: Maximum number of iterations. Defaults to None.
        :type max_iter:  list, None
        :param shuffle: Whether to shuffle samples in each iteration. Only when solver is 'sgd' 
            or 'adam'. Defaults to None.
        :type shuffle:  list, None
        :param random_state: Determines random number generation for weights and bias initialization,
            train-test split and batch sampling. Defaults to None.
        :type random_state:  list, None
        :param tol:Tolerance for stopping criteria. Defaults to None.
        :type tol:  list, None
        :param verbose: For the liblinear and lbfgs solvers set verbose to any positive
                number for verbosity.  Defaults to None.
        :type verbose: list, None
        :param warm_start: If 'True', reuse the solution of the previous call to fit
            as initialization, otherwise, just erase the previous solution. Defaults to None.
        :type warm_start:  list, None
        :param momentum: Momentum for gradient descent update. Should be between 0 and 1. 
            Defaults to None.
        :type momentum:  list, None
        :param nesterovs_momentum: Wheter to use Nesterov's momentum. Only when solves is 'sgd'
            and momentum > 0. Defaults to None.
        :type nesterovs_momentum:  list, None
        :param early_stopping: Whether to use early stopping to terminate training when 
            validation score is not improving. Defaults to None.
        :type early_stopping: list, None
        :param validation_fraction: The proportion of training data to set aside as validation 
            set fot early stopping. Must be between 0 and 1. Only if early stopping is 'True'.
            Defaults to None.
        :type validation_fraction: list, None
        :param beta_1: Exponential decay rate for estimates of first moment vector in adam, 
            should be between 0 and 1. Only used when solver is 'adam'. Defaults to None.
        :type beta_1: list, None
        :param beta_2: Exponential decay rate for estimates of second moment vector in adam, 
            should be between 0 and 1. Only used when solver is 'adam'. Defaults to None.
        :type beta_2: list, None
        :param epsilon: Value for numerical stability in adam. Only used when solver is 'adam'. 
           Defaults to None.
        :type epsilon: list, None
        :param n_iter_no_change: Maximum number of epoch to not meet tol improvement. 
            Only when solver is 'sgd' or 'adam'. Defaults to None.
        :type n_iter_no_change: list, None
        :param max_fun: Maximum number of loss functions calls. Only when solver is 'lbfgs'. 
            Defaults to None.
        :type max_fun: list, None
        """
        
        param_grid = {}
        model_mlp = skMLPClassifier()
        
        # https://stackoverflow.com/questions/2912615/how-to-iterate-over-function-arguments
        arguments = locals()
        del arguments['X_train']
        del arguments['Y_train']
        
        for arg in arguments.items():
            
            if isinstance(arg[1], list):
                param_grid[arg[0]] = arg[1]
        
        clf = GridSearchCV(model_mlp, param_grid)
        clf.fit(X_train, Y_train)
        
        best_params = clf.best_params_
        
        # https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
        self.set_model(skMLPClassifier(**best_params))