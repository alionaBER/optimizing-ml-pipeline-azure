# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

Some parts were set as the part of the task (e.g. choice of the accuracy metric in the custom model or data preparation step) and are assumed to be fixed.

## Summary

The dataset contains data about marketing campaign for bank customers, including socio-economical characteristics of the customers (e.g. maritual, employment and education status) and campaign infomation (e.g. time, campaign identifier). abcence of the definitions for the variables does not allow to evaluate there relevance to the problem and exlude possible data leakage. The target variable is not explicitly named (*y*), likely, it is the presence or absence of response to a campaign.  

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**



## Scikit-learn Pipeline

The training data is cleaned and transformed to make if sutable for training. Some categorical variables are one-hot encoded (e.g. education), others transformed to binary variables (e.g. *yes* corresponding to 1 and *no* to 0) and numeric values (e.g. days of the week). 

There are two classes of the target variable and they are highly imbalanced. The majority class constitutes 88.8% and a naive model of predicting *no* would result in 88.8% accuracy.

The custom model is a logistic regression with two hyperparameters that are tuned using HyperDrive:

- Inverse of regularization strength
- Maximum number of iterations to converge

The random search of the parameter space was chosen for its relative computational efficiency (in comparison to the Bayesian sampling) and the ability to explore the parameter space with both continuous and discrete parameters. The random sampling is compatible with the early stopping policy that has a potential of lowering computation time and costs. The perfomance metric is evaluated every time the script reports the metric and the Bandit policy is configured to terminate any training runs that are below the calculated value with the slack factor of 0.15 (see details in [the documentation](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)). A number of runs was configured to 12.

The best model achived the accuracy of 91.02%. The hyperparameter values of this moded: the inverse of regularization strength of 6.167 and the maximum number of iterations to converge of 200. 

## AutoML

A voting ensemble achieved the highest accuracy of 91.66%. There were ten estimators in the enseble, but the fact that it was an ensemble conceales individual hyperparameters. 

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**



## Future work

The training on the highly unbalanced dataset is likely to bias the accuracy results upwards. Both modelling approches only gained a couple of percent of accuracy in comparison to the naive model (no model). Measures to counteract the unbalanced training dataset are recommended for future experiments.

The best custom model's inverse of regularization strength was the smallest amonng randomly sampled in the parameter space. Possible improvement steps include cupping the search space with a smaller value from the top to explore smaller values of this hyperparameter or increasing the number of runs.

![](custom_model_hyperparameters.PNG "Custom model hyperparameters")

## Proof of cluster clean up
Image of cluster marked for deletion:

![](compute_cluster_deletion.PNG "Deleted compute cluster")

