"""
Computation of the measure 'AUC Mu'. This measure requires installation of the
numpy, Apache Spark and pyspark interpreter.
This code corresponds to the paper: Kleiman, R., Page, D. ``AUC Mu: A 
Performance Metric for Multi-Class Machine Learning Models``, Proceedings of the
2019 International Conference on Machine Learning (ICML).
"""

__author__ = "PoWeiHuang"
__copyright__ = "Copyright 2019"
__credits__ = ["PoWeiHuang"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "PoWeiHuang"
__email__ = "poweihuang@cathayholdings.com.tw"
__status__ = "Production"

import numpy as np
from pyspark.mllib.linalg import Vector, VectorUDT, DenseVector 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf, DataFrame, avg

#----------------------------------------------------------------------
def pyspark_auc_mu(data, y_true, y_score, A=None, W=None):
    """
    Compute the multi-class measure AUC Mu from prediction scores and labels from Spark dataframe.
    
    Parameters
    ----------
    data : Spark dataframe
        The prediction output table from pyspark.ml
        
    y_true : double type column in data, shape = [n_samples]
        The true class labels column from spark dataframe in the range [0, n_samples-1]
    
    y_score : vector type column in data, shape = [n_samples, n_classes]
        Target scores, where each element in a vector is a categorical distribution over the 
        n_classes.
    
    A : array, shape = [n_classes, n_classes], optional
        The partition (or misclassification cost) matrix. If ``None`` A is the
        argmax partition matrix. Entry A_{i,j} is the cost of classifying an
        instance as class i when the true class is j. It is expected that
        diagonal entries in A are zero and off-diagonal entries are positive.
    
    W : array, shape = [n_classes, n_classes], optional
        The weight matrix for incorporating class skew into AUC Mu. If ``None``,
        the standard AUC Mu is calculated. If W is specified, it is expected to 
        be a lower triangular matrix where entrix W_{i,j} is a positive float
        from 0 to 1 for the partial score between classes i and j. Entries not
        in the lower triangular portion of W must be 0 and the sum of all 
        entries in W must be 1.
    
    Returns
    -------
    auc_mu : float
    
    References
    ----------
    .. [1] Kleiman, R., Page, D. ``AUC Mu: A Performance Metric for Multi-Class
           Machine Learning Models``, Proceedings of the 2019 International
           Conference on Machine Learning (ICML).    
       [2] https://github.com/kleimanr/auc_mu
    """
    
    n_classes = data.select(y_true).distinct().count()
    n_samples = data.select(y_score).count()
    # Validate input arguments
    if not isinstance(data, DataFrame):
        raise TypeError("Expected data to be DataFrame, got: %s"
                        % type(data))     
    if not data.select(y_true).dtypes[0][1] == 'double':
        raise TypeError("Expected column y_true to be double, got: %s"
                        % data.select(y_true).dtypes[0][1])
    if not data.select(y_score).dtypes[0][1] == 'vector':
        raise TypeError("Expected column y_score to be vector, got: %s"
                        % data.select(y_true).dtypes[0][1])
    if not data.select(y_true).count() == n_samples:
        raise ValueError("Expected y_true to be shape %s, got: %s"
                        %(str(data.select(y_score).count()), str(data.select(y_true).count())))
    
    slen = udf(lambda s: len(s), IntegerType())    
    if not data.select(slen(col(y_score))).groupBy().avg().collect()[0][0] == n_classes:
        raise ValueError("Expected y_true values in range 0..%i, got: %s"
                        %(n_classes-1, str(data.select(slen(data.y_score)).groupBy().avg().collect()[0][0])))        
    if A is None:
        A = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    if not isinstance(A, np.ndarray):
        raise TypeError("Expected A to be np.ndarray, got: %s" 
                        % type(A))
    if not A.ndim == 2:
        raise ValueError("Expected A to be 2 dimensional, got: %s"
                         % A.ndim)
    if not A.shape == (n_classes, n_classes):
        raise ValueError("Expected A to be shape (%i, %i), got: %s"
                         %(n_classes, n_classes, str(A.shape)))
    if not np.all(A.diagonal() == np.zeros(n_classes)):
        raise ValueError("Expected A to be zero on the diagonals")
    if not np.all(A >= 0):
        raise ValueError("Expected A to be non-negative")
    
    if W is None:
        W = np.tri(n_classes, k=-1)
        W /= W.sum()
    if not isinstance(W, np.ndarray):
        raise TypeError("Expected W to be np.ndarray, got: %s" 
                        % type(W))
    if not W.ndim == 2:
        raise ValueError("Expected W to be 2 dimensional, got: %s"
                         % W.ndim)
    if not W.shape == (n_classes, n_classes):
        raise ValueError("Expected W to be shape (%i, %i), got: %s"
                         %(n_classes, n_classes, str(W.shape)))
    
    auc_total = 0.0
    for class_i in xrange(n_classes):
        preds_i = data.select(y_score).where(col(y_true) == class_i)
        n_i = preds_i.count()
        
        for class_j in xrange(class_i):
            preds_j = data.select(y_score).where(col(y_true) == class_j)
            temp_preds = preds_i.union(preds_j)
            
            n_j = preds_j.count()
            n = n_i+n_j
            #temp_preds: concat prob vectors which class = i and j
            temp_preds = DenseVector(temp_preds.select("probability").rdd.map(lambda x: x[0]).collect())
            
            #temp_labels: convert the two selected classes to a binary class vector
            temp_labels = np.zeros((n), dtype=int)
            temp_labels[n_i:n] = 1
            
            # v: differencing by vector_{i,.} and vector_{j,.} in partition matrix
            v = A[class_i, :] - A[class_j, :]       
            score = temp_preds.dot(v)
            df = np.column_stack([score, temp_labels])
            concat_df = map(lambda x: (float(x[0]), float(x[1:])), df)
            auc_mu_df = spark.createDataFrame(concat_df,schema=["score", "temp_label"])
            
            evaluator = BinaryClassificationEvaluator(labelCol="temp_label", rawPredictionCol="score", metricName='areaUnderROC')
            score_i_j = evaluator.evaluate(auc_mu_df)
            auc_total += W[class_i, class_j]*score_i_j

            
    return auc_total
