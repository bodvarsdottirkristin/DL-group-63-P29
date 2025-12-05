# DL-group-63-P29
# Trajectory Prediction of Cargo Vessels Around Denmark

This repository contains the code for our project in **02456 Deep Learning (DTU, Fall 2025)**.
The goal is to predict short-term trajectories of cargo ships operating in Danish waters using AIS data and deep learning.

We follow a pipeline inspired by Murray & Perera (2021) that combines:

- Recurrent Autoencoders (RAE) for trajectory representation learning
- Density-based clustering (HDBSCAN) in the laten space
- A GRU-based calssifier to predict behavioural clusters
- Cluster-conditioned GRU models for short-term trajectory forecasting

To replicate our results, notebooks should be run in the following order:
1. src/data_processing/data_pipeline.ipynb
2. src/models/train_RAE.ipynb
3. src/models/train_HDBSCAN.ipynb
4. src/models/train_classifier.ipynb
5. src/models/train_prediction_rnn.ipynb
6. src/models/final_testing.ipynb

## Project overview

Maritime AIS (Automatic Identification System) data provides irregular, noisy observations of vessel positions and motion. 
Our aim is to:

-> Predict the next **30 minutes** of a vessel's trajectory given the past **30 minutes** of AIS observations.

To make this feasable, we:

1. Preprocess raw AIS messages into clean, uniform trajectories
2. Learn latent representations with a Recurrent Autoencoder
3. Cluster trajectories in latent space to discover behavioural groups
4. Train:
   - a classifier that predicts the future cluster from the past segment
   - and a cluster-specific GRU predictor for the future trajectory
  
This repository container the full pipeline implementation in **PyTorch** 


## Repository structure
[Structure of the repo here]


# info on the original AIS dataset
Columns in *.csv file                   Format
----------------------------------------------------------------------------------------------------------------------------------------------------
1.    Timestamp                         Timestamp from the AIS basestation, format: 31/12/2015 23:59:59 
2.    Type of mobile                    Describes what type of target this message is received from (class A AIS Vessel, Class B AIS vessel, etc)
3.    MMSI                              MMSI number of vessel
4.    Latitude                          Latitude of message report (e.g. 57,8794)
5.    Longitude                         Longitude of message report (e.g. 17,9125)
6.    Navigational status               Navigational status from AIS message if available, e.g.: 'Engaged in fishing', 'Under way using engine', mv.
7.    ROT                               Rot of turn from AIS message if available
8.    SOG                               Speed over ground from AIS message if available
9.    COG                               Course over ground from AIS message if available
10.   Heading                           Heading from AIS message if available
11.   IMO                               IMO number of the vessel
12.   Callsign                          Callsign of the vessel 
13.   Name                              Name of the vessel
14.   Ship type                         Describes the AIS ship type of this vessel 
15.   Cargo type                        Type of cargo from the AIS message 
16.   Width                             Width of the vessel
17.   Length                            Lenght of the vessel 
18.   Type of position fixing device    Type of positional fixing device from the AIS message 
19.   Draught                           Draugth field from AIS message
20.   Destination                       Destination from AIS message
21.   ETA                               Estimated Time of Arrival, if available  
22.   Data source type                  Data source type, e.g. AIS
23.   Size A                            Length from GPS to the bow
24.   Size B                            Length from GPS to the stern
25.   Size C                            Length from GPS to starboard side
26.   Size D                            Length from GPS to port side

