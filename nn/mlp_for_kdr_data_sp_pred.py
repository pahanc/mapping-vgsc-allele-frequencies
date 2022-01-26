#Author: Penelope A. Hancock
#Python script for generating out-of-sample predictions of Vgsc allele frequencies for ten test sets
#using a multilayer perceptron neural network model. The fitted model parameters were obtained by 
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


#set the hyperparameters of the multilayer perceptron model using the parameter values chosen using model tuning
tune_run=13
batch_size_vec=[5000,20000,40000]
learning_rate_vec=[5e-04,2e-04,1e-04]
dropout_vec=[0,0.1,0.3]
tun_params_grd=np.array([(batch_size1,learning_rate1,dropout1) for batch_size1 in batch_size_vec for learning_rate1 in learning_rate_vec for dropout1 in dropout_vec])


def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(100,input_dim=dim,kernel_constraint=nonneg()))
    model.add(ELU())
    model.add(Dropout(tun_params_grd[tune_run,2]))
    model.add(BatchNormalization())
    return model


import feather
import os

#import the Vgsc allele occurrence data
target_df=feather.read_dataframe("kdr_df_sp.feather")
kdr_target = np.asarray(target_df)
kdr_target = to_categorical(kdr_target,num_classes=3)

#import the sampling year data
start_yr_df=feather.read_dataframe("start_yr_df_sp.feather")
start_yr = np.asarray(start_yr_df)
start_yr = start_yr-1

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
    #Make the training and tests data sets    
    X_train_met,X_test_met=start_yr_species_feat[train_inds.astype(int)],start_yr_species_feat[test_inds.astype(int)]    
    y_train,y_test=kdr_target[train_inds.astype(int)],kdr_target[test_inds.astype(int)]
    #Define the model
    mlp = create_mlp(X_train_met.shape[1])
    x = Dense(3, activation="softmax")(mlp.output)
    model = Model(inputs=mlp.input, outputs=x)
    opt = Adam(lr=tun_params_grd[tune_run,1], decay=1e-6)
    model.compile(optimizer=opt,loss='categorical_crossentropy')
    #Load the model weights obtained by hyperparameter tuning
    model.load_weights("mdl_wts_mlp13.hdf6")
    model.compile(optimizer=opt,loss='categorical_crossentropy')
    #Run the model to predict allele frequencies in the test data set
    Y_pred2 = model.predict(X_test_met)
    #Save the output
    fname = 'ypred_mlp'+str(k)+'.txt'
    with open(fname,'w') as fid:
        np.savetxt(fid,Y_pred2)



