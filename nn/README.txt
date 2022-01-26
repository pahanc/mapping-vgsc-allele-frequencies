Python Code for running the multilayer perceptron neural network (NN) modelling analysis described in:

"Hancock, P.A., Lynd, A., Wiebe, A., Devine, M., Essandoh, J., Wat'senga, F., Manzambi, E., Agossa, F., Donnelly, M.J., Weetman, D., Moyes, C.L., 2022, Modelling spatiotemporal trends in the frequency of genetic mutations conferring insecticide target-site resistance in African malaria vector species", BMC Biology"

The NN model generates predictions on a test set containing frequencies of Vgsc insecticide resistance mutations using a set of predictor variables.

Instructions for reproducing the 10-fold out-of-sample validation results for the NN model for all 10 test sets are as follows:

Run the following from the command line:

python mlp_for_kdr_data_sp_pred.py

This script requires the following the input following files:
1. " mlp_for_kdr_data_sp_pred.py": function for formatting the input data andgenerating predictions using the neural network model
2. "kdr_df_sp.feather": Data set containing all labels (Vgsc allele presence) for each mosquito
3. "features_df_sp.feather": Data set with all the features (predictor variables) corresponding to each label
4. "species_df.feather": Data set indicating the species of each mosquito tested
