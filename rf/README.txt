Python Code for running the random forest (RF) modelling analysis described in:

"Hancock, P.A., Lynd, A., Wiebe, A., Devine, M., Essandoh, J., Wat'senga, F., Manzambi, E., Agossa, F., Donnelly, M.J., Weetman, D., Moyes, C.L., 2022, Modelling spatiotemporal trends in the frequency of genetic mutations conferring insecticide target-site resistance in African malaria vector species", BMC Biology"

The RF model generates predictions on a test set containing frequencies of Vgsc insecticide resistance mutations using a set of predictor variables.

Instructions for reproducing the 10-fold out-of-sample validation results for the RF model for all 10 test sets are as follows:

Run the following from the command line:

python random_forest_py_pred.py

This script requires the following the input following files:
1. "random_forest_py_pred.py": function for formatting the input data and training the random forest model
2. "kdr_df_sp.feather": Data set containing all labels (Vgsc allele presence) for each mosquito
3. "features_df_sp.feather": Data set with all the features (predictor variables) corresponding to each label
4. "species_df.feather": Data set indicating the species of each mosquito tested
5. "start_year_df_sp.feather": Data set indicating the sampling year for each data point
6. "train_indsX.feather" where X=1...10: Files containing the indices of each data point in training set X
7. "test_indsX.feather" where X=1...10: Files containing the indices of each data point in test set X
