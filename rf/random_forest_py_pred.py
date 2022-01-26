#Author: Penelope A. Hancock
#Python script for generating out-of-sample predictions of Vgsc allele frequencies for ten test sets
#using a random forest model. The fitted model parameters were obtained by 
#hyperparameter tuning.

import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from random import seed
from random import randint

# --- sklearn
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier

# --- keras
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras import optimizers
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from keras.callbacks import ModelCheckpoint
from keras import models
import keras.backend as K

import feather
import os

#import the Vgsc allele occurrence data
target_df=feather.read_dataframe("kdr_df_sp.feather")
kdr_target = np.asarray(target_df)

#import the sampling year data
start_yr_df=feather.read_dataframe("start_yr_df_sp.feather")
start_yr = np.asarray(start_yr_df)

start_yr= to_categorical(start_yr)

#import the species id data
species_df=feather.read_dataframe("species_df.feather")
species = np.asarray(species_df)

species= to_categorical(species,num_classes=4)

start_yr_species = np.concatenate((start_yr,species),axis=1)

#import the environmental predictor variables
features_df=feather.read_dataframe("features_df_sp.feather")
features=np.asarray(features_df)

#join all the features into one data array
start_yr_species_feat = np.concatenate((start_yr_species,features),axis=1)


n_samples=len(kdr_target)
indices = np.arange(n_samples)

#set the hyperparameters of the random forest model using the parameter values chosen using model tuning
max_features_vec=[50,100,150]
min_samples_leaf_vec=[5,10,20]
n_estimators_vec=[10,100,500]
tun_params_grd=np.array([(max_features1,min_samples_leaf1,n_estimators1) for max_features1 in max_features_vec for min_samples_leaf1 in min_samples_leaf_vec for n_estimators1 in n_estimators_vec])
tune_run=8

##K fold validation sets
for k in range(1,11):
    tr_fname = 'train_inds'+str(k)+'.feather'
    train_inds_df=feather.read_dataframe(tr_fname)
    train_inds = np.asarray(train_inds_df)
    train_inds = train_inds.reshape(len(train_inds))
    tst_fname = 'test_inds'+str(k)+'.feather'
    test_inds_df=feather.read_dataframe(tst_fname)
    test_inds = np.asarray(test_inds_df)
    test_inds = test_inds.reshape(len(test_inds))
    #Make the training and test data sets
    X_train_met,X_test_met=start_yr_species_feat[train_inds.astype(int)],start_yr_species_feat[test_inds.astype(int)]    
    y_train,y_test=kdr_target[train_inds.astype(int)],kdr_target[test_inds.astype(int)]
    #Define the random forest model
    model = RandomForestClassifier(max_features=tun_params_grd[tune_run,0],min_samples_leaf=tun_params_grd[tune_run,1],n_estimators=tun_params_grd[tune_run,2])
    #Fit the model to the training data set
    model.fit(X_train_met,y_train.ravel())
    #Predict the frequencies of each allele in the test data set
    Y_pred2 = model.predict_proba(X_test_met)
    #Save the output
    fname = 'rfpred'+str(k)+'.txt'
    with open(fname,'w') as fid:
        np.savetxt(fid,Y_pred2)
 



