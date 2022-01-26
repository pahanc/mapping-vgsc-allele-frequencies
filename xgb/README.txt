R Code for running the Extreme Gradient Boosting (XGB) modelling analysis described in:

"Hancock, P.A., Lynd, A., Wiebe, A., Devine, M., Essandoh, J., Wat'senga, F., Manzambi, E., Agossa, F., Donnelly, M.J., Weetman, D., Moyes, C.L., 2022, Modelling spatiotemporal trends in the frequency of genetic mutations conferring insecticide target-site resistance in African malaria vector species", BMC Biology"

The XGB model generates predictions on a test set containing frequencies of Vgsc insecticide resistance mutations using a set of predictor variables.

Instructions for reproducing the 10-fold out-of-sample inner validation results for the XGB model for a given outer validation set (from a set of 10 outer validation sets) are as follows:

For test set 2, run the following in R:

source("l0_pred_fn_val_xgb_ind.r")
pred_runs_val_xgb_ind.r(2)

This script requires the following the input following files:
1. "inputs_data_all_wa_ea_sp3.r": Data set with all the labels (Vgsc allele presence) and features (predictor variables)
2. "stk_val_inds5.r": indices of data points to withhold for the 10 test sets
3. "l0_pred_fn_val_xgb_ind.r": function for formatting the input data and training the xgboost model
