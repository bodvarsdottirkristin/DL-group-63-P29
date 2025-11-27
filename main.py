import pandas as pd
import numpy as np
import random
import wandb

#from src.datacleaning import fn, fn_get_dk_ports
from src.datasets.window_maker import make_past_future_windows, load_parquet_files
from src.models.classification_rnn import ClassificationRNN
from src.training.train_classification_rnn import run_classification_train_rnn
from src.training.train_predictor_rnn import *



def main():

    # make_past_future_windows(
    #     past_min=30,future_min=30,
    #     input_path="data/aisdk/processed/aisdk_2025", 
    #     output_path="data/aisdk/processed/windows_30_30")
    
    # df = pd.read_parquet("data/aisdk/processed/windows_30_30/cluster_id=0")
    # print(df.head())

    # print(df.info())
    #load_windows_from_parquet("data/aisdk/processed/windows_30_30/cluster_id=0")

    X, Y, C = load_parquet_files()

    #run_classification_train_rnn(X, C, False)

    run_predictor_cluster(X, Y, C, 1, False)

    

if __name__ == "__main__":
    main()
    