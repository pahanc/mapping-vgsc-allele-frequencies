# mapping-vgsc-allele-frequencies
This directory contains code for running stacked generalisation geospatial modelling analyses described in :

"Hancock, P.A., Lynd, A., Wiebe, A., Devine, M., Essandoh, J., Wat'senga, F., Manzambi, E., Agossa, F., Donnelly, M.J., Weetman, D., Moyes, C.L., 2022, Modelling spatiotemporal trends in the frequency of genetic mutations conferring insecticide target-site resistance in African malaria vector species", BMC Biology"

The method performs spatiotemporal prediction of the frequencies of three Vgsc alleles in mosquito samples: Vgsc-995L, Vgsc-995F and Vgsc-995S. Mosquito samples contain the following species from the Anopheles gambiae complex: An. gambiae, An. coluzzii, An. arabiensis.

Individual directories contain the code for running the level-0 machine-learning models, including the extreme gradient boosting (xgb), random forest (rf) and neural network (nn) models, and the multinomial meta-model.

System requirements: Code for runnning the xgb and inla models is written in R software, which runs on a wide variety of UNIX platforms, Windows and Mac OS. The R software is open source and quick to install. Code for running the rf and mlp models is written in open source Python software, installed using the Anaconda platform.

Software requirements: R, using the following packages: R-INLA LaplacesDemon zoo mboost xgboost data.table caret

The R code has been tested on R version 3.5.0 with the package R-INLA version 17.06.2, the package xgboost version 0.71.2, and the package data.table version 1.12.8.

The Python code has been tested on python version 3.6.8 with the package keras version 2.2.4 and the package sklearn version 0.20.3.
