# spark-AUCμ
AUCμ is a performance matric for multi-class classification models and it is an extension of the standard two-class area under the receiver operating characteristic curve (AUC-ROC) written by Ross Kleiman. **This repo produced the origin source code of AUCμ from python local version to distributed machine on Apache Spark**.

# Getting Started
These instructions provide the matric on distributed machine for development and testing purpose.

## Prerequisites
The developer version has the following requirements: 
* Python
* Spark 2.3.1. Spark may be downloaded from the [Spark website](https://spark.apache.org/). In order to use this package, you need to use the pyspark interpreter or another Spark-compliant python interpreter. See the [Spark guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html) for more details.
* numpy

## Installing
Simply place spark_auc_mu.py in any directory in your Python path.

## Usage
Example Usage:

    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    data = spark.createDataFrame(pd.read_csv('iris.csv', header=0,\
    names=['sepal_length','sepal_width','petal_width','petal_length','variety']))
    inputCols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    vecAssembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    va = vecAssembler.transform(data)
    stringIndexer = StringIndexer(inputCol="variety", outputCol="variety_index")
    si_model = stringIndexer.fit(va)
    td = si_model.transform(va)
    (trainData, testData) = td.randomSplit([0.8, 0.2])
    rf = RandomForestClassifier(numTrees=2, maxDepth=1, labelCol="variety_index", seed=42)
    model = rf.fit(trainData)
    predictions = model.transform(testData)
    
    # Input transformed dataframe, true label column and predicted probability column to get the matric directory.
    # A and W are optional parameters to control the costs of partition matrix and skew data punishment coefficient.
    auc_mu = pyspark_auc_mu(predictions, "variety_index", "probability", A=None, W=None)
    >>>0.8333333333333333
    
Additional information regarding use of an alternative partition matrix or weight matrix is contained in the auc_mu.auc_mu Docstring.

# Authors
* PoWeiHuang

# Reference
* [AUCµ: A Performance Metric for Multi-Class Machine Learning Models](http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf)
* [AUCµ github repo](https://github.com/kleimanr/auc_mu)

# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/poweihuang/spark-aucmu/blob/master/LICENSE) files for details
