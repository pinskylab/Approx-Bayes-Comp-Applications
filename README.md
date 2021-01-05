# Using stage structure to infer process-based models for species on the move

### Jude D. Kong, Emily Moberg, Becca Selden, Alexa Fredston, Malin Pinsky
## Overview

This repository contains Python scripts for estimating parameters of models (of varying complexity) using an Approximate Bayesian Computation (ABC) framework. 

The repository is organized as follows:


* **scripts:** contains all code for testing and validating our framework. In particular: 

     * *GAM_Process_based_comparison*:  contain Python scripts for comparing stage-structured process based-model  to  generalized additive models (GAM) and process-based models without stage structure. 

     * *bias_param_est_species1*: contain Python scripts for estimating and plotting the bias in parameter estimation for 100 different species of life history type 2 (see  the manuscript for detailed information).
     
     * *bias_param_est_species2*: contain Python scripts for estimating and plotting the bias in parameter estimation for 100 different species of life history type 2 (see  the manuscript for detailed information).
     
     * *bias_param_est_species3*: contain Python scripts for estimating and plotting the bias in parameter estimation for 100 different species of life history type 3 (see  the manuscript for detailed information).
     
     * *projection_bias_species1*: contain Python scripts for estimating and plotting the bias in abundance of 100 replicates of species from the first virtual case study without process noise (see  the manuscript for detailed information).
     
     * *projection_bias_species2*: contain Python scripts for estimating and plotting the bias in abundance of 100 replicates of species from the second virtual case study without process noise (see  the manuscript for detailed information).
     
     * *projection_bias_species3*: contain Python scripts for estimating and plotting the bias in abundance of 100 replicates of species from the third virtual case study without process noise (see  the manuscript for detailed information).
     
     * *impact_process_noise_projection*: contain Python scripts for estimating and plotting the bias in abundance of 100 replicates of species from the second virtual case study with process noise (see  the manuscript for detailed information).
     
     * *impact_species_misspecification*: contain Python scripts for estimating and plotting the bias in abundance of 100 different species with life history type 2 calibrated using data from species with life history type 3(see  the manuscript for detailed information).
     

* **data:** contains  two .csv file: jude_cod_test.csv and jude_cod_train.csv.  
    * *jude_cod_test.csv*  contains data on gadus morhua_Atl haulid, region where it was caught, year, month stratum, latitude, Ion, depth, bottom temperature and length class from 1968 to 2000. 
   
    * *jude_cod_test.csv** contains data on gadus morhua_Atl haulid, region where it was caught, year, month stratum, latitude, Ion, depth, bottom temperature and length class from 2001 to 2015. This data is used to further validate our framework. 


## Computational requirements

Analyses were conducted in RStudio 1.2.5033 on a MacBook Pro (16-inch, 2019)  with the following specifications: Processor 2.6 GHz 6-Core Intel Core i7, 
Memory 16 GB 2667 MHz DDR4, Startup Disk Macintosh HD, Graphics Intel UHD Graphics 630 1536 MB.


## Use, problems, and feedback

If you encounter any errors or problems, please create an Issue here. Likewise, please consider starting an Issue for any questions so that others can view conversations about the analysis and code. Again, don't hesitate to contact us at jdkong@yorku.ca

