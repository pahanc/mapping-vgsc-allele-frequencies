#Author: Penelope A. Hancock
#R script for generating out-of-sample predictions of Vgsc allele frequencies for ten test sets
#using an extreme gradient boosting model. The fitted model parameters were obtained by 
#hyperparameter tuning.


pred_val_xgb<-function(val_run){
#"pred_val_xgb" calculates out-of-sample predictions for an extreme gradient boosting model that has
# been tuned on the Vgsc allele frequency data set "kdr_data_wa_ea_long". Indices out the test
# set for each of 10 validation runs (val_run) are contained in "stk_val_inds_kdr_wa_ea_sp3.r"
  
#XGB
library(data.table)
library(xgboost)

val_run<-as.integer(as.numeric(val_run))
  
load("inputs_data_kdr_wa_ea_sp3.r") #Load the data set
data_all<-inputs$kdr_data_wa_ea_long
load("stk_val_inds_kdr_wa_ea_sp3.r") #Load the indices of the test sets
stk_val_inds_i<-stk_val_inds[[val_run]]
groups<-inputs$groups # "groups" is an index variable indicating which sample each record in
#kdr_data_wa_ea_long belongs to
inds<-which(is.element(groups,stk_val_inds_i))
stk_val_inds_i<-inds
test_inds<-stk_val_inds_i


xgb_params<-list() # "xgb_params" contains the hyper parameters of the extreme gradient boosting model
xgb_params$objective<-"multi:softprob"
xgb_params$eval_metric<-"mlogloss"
xgb_params$num_class<-3
xgb_params$max_depth<-8
xgb_params$min_child_weight<-10
xgb_params$subsample<-1

nround<-20 #The number of boosted trees being fitted
                   
xgbPredJ<-list()

testJ<-data_all[test_inds,]
trainJ<-data_all[-test_inds,]
trainJ<-data.matrix(trainJ)
response<-trainJ[,"resp"]
covariates<-as.matrix(trainJ[,inputs$covariate.names.all.wa.ea])
colnames(covariates)<-inputs$covariate.names.all.wa.ea
trainJ<-xgb.DMatrix(data=covariates,label=response) #The training data set

#Fit the extreme gradient boosting model to the training data set
xgbFitJ<-xgb.train(params=xgb_params,data = trainJ,nrounds=nround,verbose = TRUE,prediction=TRUE)

response_p<-testJ[,"resp"]
covariates_p<-as.matrix(testJ[,inputs$covariate.names.all.wa.ea])
testJ<-xgb.DMatrix(data=covariates_p,label=response_p) #The test data set
xgbPredJ<-predict(xgbFitJ,testJ,ntree_limit=50) #Use the fitted model
#to generate predictions for the data point in the test data set

#Save the output
output<-list()
output$xgbFitJ<-xgbFitJ
output$xgbPredJ<-xgbPredJ
output$data_all<-data_all
output$test_inds<-test_inds
filename<-paste("xgbPred-current",val_run,".r",sep="")
save(output,file=filename)


}#end function
