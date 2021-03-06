#Author: Penelope A. Hancock
#Script for fitting a multinomial logit regression model to 
#out-of-sample prediction of Vgsc allele frequencies

load("inputs_data_kdr_wa_ea_sp3.r")#load the Vgsc allele frequency data
load("stk_val_inds_kdr_wa_ea_sp3.r")#load the indices of the 10 test data sets
source("load_gen.r")#functions needed to run the script
inla_preds_val<-list()
res_list<-list()
prob.est.lst<-list()
for (val_ind in 1:10){#for ten test sets
  print(paste("val_ind", val_ind,sep=" "))
  test_inds<-stk_val_inds[[val_ind]]
  data_all<-inputs$kdr_data_wa_ea
  #insert NAs in place of data values in rows indexed as part of the out-of-sample test sets
  data_all[test_inds,c("l1014l","l1014f","l1014s")]<-c(NA,NA,NA)
  #Make a vector of the numbers of each allele in each sample
  kdr_counts<-round(data_all[,"tot_no_mosq"]*data_all[,c("l1014l","l1014f","l1014s")]/100)
  kdr_counts_vec<-NULL
  for (i in 1:nrow(kdr_counts)) {kdr_counts_vec<-c(kdr_counts_vec,as.numeric(kdr_counts[i,]))}
  kdr_counts_vec<-as.data.frame(kdr_counts_vec)
  colnames(kdr_counts_vec)="kdr"
  #Make an index grouping the data by sample ID
  idx<-rep(1:nrow(data_all),times=rep(3,nrow(data_all)))
  idx<-data.frame(idx)
  #Make an index grouping the data by the type of allele (l1014l,l1014f,l1014s)
  jdx<-rep(c(1,2,3),times=nrow(data_all))
  jdx2<-jdx
  jdx3<-jdx
  jdx<-data.frame(jdx)
  jdx2<-data.frame(jdx2)
  jdx3<-data.frame(jdx3)
  #Load the out-of-sample predicted allele frequencies generated by the neural network model
  #and re-order them to the ordering of the sampling data
  #and store them as the first covariate
  load("nn_oos_preds3.RData")
  covs1<-NULL
  all_preds_nn_mat<-nn_oos_preds3$preds2_all
  all_preds_nn_mat<-all_preds_nn_mat[order(nn_oos_preds3$data_all_inds),]
  for (j in 1:nrow(all_preds_nn_mat)){
    covs1<-c(covs1,all_preds_nn_mat[j,])
  }
  covs1<-unlist(covs1)
  #Load the out-of-sample predicted allele frequencies generated by the extreme gradient boosting model
  #and re-order them to the ordering of the sampling data
  #and store them as the second covariate
  load("xgb_oos_preds3.RData")
  covs2<-NULL
  all_preds_xgb_mat<-xgb_oos_preds3$preds2_all
  all_preds_xgb_mat<-all_preds_xgb_mat[order(xgb_oos_preds3$data_all_inds),]	
  for (j in 1:nrow(all_preds_xgb_mat)){
    covs2<-c(covs2,all_preds_xgb_mat[j,])
  }
  covs2<-unlist(covs2)
  #Load the out-of-sample predicted allele frequencies generated by the random forest model
  #and re-order them to the ordering of the sampling data
  #and store them as the third covariate
  load("rf_oos_preds3.RData")
  covs3<-NULL
  all_preds_rf_mat<-rf_oos_preds3$preds2_all
  all_preds_rf_mat<-all_preds_rf_mat[order(rf_oos_preds3$data_all_inds),]	
  for (j in 1:nrow(all_preds_rf_mat)){
    covs3<-c(covs3,all_preds_rf_mat[j,])
  }
  covs3<-unlist(covs3)
  
  #Transform the covariates using the empirical logit transformation
  covs1<-emplogit2(100*covs1,100)
  covs2<-emplogit2(100*covs2,100)
  covs3<-emplogit2(100*covs3,100)
  covs_mat<-cbind(covs1,covs2,covs3)
  rownames(covs_mat)<-NULL
  
  #Extract the values of the covariates for each type of Vgsc allele
  covs1.L<-covs1*(jdx[[1]]==1)
  covs1.F<-covs1*(jdx[[1]]==2)
  covs1.S<-covs1*(jdx[[1]]==3)
  covs2.L<-covs2*(jdx2[[1]]==1)
  covs2.F<-covs2*(jdx2[[1]]==2)
  covs2.S<-covs2*(jdx2[[1]]==3)
  covs3.L<-covs3*(jdx3[[1]]==1)
  covs3.F<-covs3*(jdx3[[1]]==2)
  covs3.S<-covs3*(jdx3[[1]]==3)
  
  covs_mat=cbind(covs1.L,covs1.F,covs1.S,covs2.L,covs2.F,covs2.S,covs3.L,covs3.F,covs3.S)
  
  covariate.names1<-c("covs1.L","covs1.F","covs1.S","covs2.L","covs2.F","covs2.S",
                      "covs3.L","covs3.F","covs3.S")
  
  #Write a formula for the predicted allele frequencies with the covariates as fixed effects
  #and an iid random effect for each sample
  formula = as.formula(paste("kdr ~ -1 +", paste(c(covariate.names1),collapse='+'), " + ",
                             "f(idx, model='iid',hyper = list(prec = list(initial = log(0.00001),fixed = TRUE)))")) 
  #Run the INLA model
  r = inla(formula,family = "poisson",data = data.frame(kdr_counts_vec, idx, jdx, jdx2,jdx3, covs_mat),control.compute = list(config=TRUE),control.predictor = list(compute=TRUE))
  #Store the results for each test set
  res_list[[val_ind]]<-r
  
  #Calculate the out-of-sample predictions by sampling from the posterior
  #and using the multinomial coefficients to estimate the probabilities
  nsample = 100
  xx = inla.posterior.sample(nsample, r)
  ## (or use target = n*m+1:m)
  prob.est<-matrix(nrow=2379,ncol=3)
  kk=1
  for (k in 1:2379){
    target = rownames(xx[[1]]$latent)[c(kk,kk+1,kk+2)]
    prob = matrix(NA, nsample, 3)
    for (i in 1:nsample) {
      eta = xx[[i]]$latent[target, 1]
      prob[i, ] = exp(eta) / sum(exp(eta))
    }
    prob.est[k,] <- apply(prob,2,mean,na.rm=T)
    kk=kk+3
  }
  prob.est.lst[[val_ind]]<-prob.est
}#End of loop over test sets

#Make a list of the predictions for each test set
inla_preds_val<-list()
for (i in 1:10) inla_preds_val[[i]]<-prob.est.lst[[i]][stk_val_inds[[i]],]

#Extract the predictions for each type of allele (l1014l,l1014f,l1014s)
inla_preds_val1<-NULL
for (i in 1:10) inla_preds_val1<-c(inla_preds_val1,inla_preds_val[[i]][,1])
inla_preds_val2<-NULL
for (i in 1:10) inla_preds_val2<-c(inla_preds_val2,inla_preds_val[[i]][,2])
inla_preds_val3<-NULL
for (i in 1:10) inla_preds_val3<-c(inla_preds_val3,inla_preds_val[[i]][,3])
stk_val_inds_all<-unlist(stk_val_inds)

#Calculate the out-of-sample RMSE across all allele frequencies
sqrt(mean((c(inla_preds_val1,inla_preds_val2,inla_preds_val3)-
             c(inputs$kdr_data_wa_ea[stk_val_inds_all,"l1014l"]/100,
               inputs$kdr_data_wa_ea[stk_val_inds_all,"l1014f"]/100,
               inputs$kdr_data_wa_ea[stk_val_inds_all,"l1014s"]/100))^2))

