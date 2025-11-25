import pandas as pd
import numpy as np
import random
import wandb

#from src.datacleaning import fn, fn_get_dk_ports
from src.datasets.window_maker import make_past_future_windows



def main():

    # make_past_future_windows(
    #     past_min=30,future_min=30,
    #     input_path="data/aisdk/processed/aisdk_2025", 
    #     output_path="data/aisdk/processed/windows_30_30")
    
    # Start a new wandb run to track this script.
    # run = wandb.init(
    #     # Set the wandb entity where your project will be logged (generally your team name).
    #     entity="ais-maritime-data",
    #     # Set the wandb project where this run will be logged.
    #     project="ais-maritime-prediction",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": 0.02,
    #         "architecture": "CNN",
    #         "dataset": "CIFAR-100",
    #         "epochs": 10,
    #     },
    # )

    # Simulate training.
    # epochs = 10
    # offset = random.random() / 5
    # for epoch in range(2, epochs):
    #     acc = 1 - 2**-epoch - random.random() / epoch - offset
    #     loss = 2**-epoch + random.random() / epoch + offset

    #     # Log metrics to wandb.
    #     run.log({"acc": acc, "loss": loss})

    # # Finish the run and upload any remaining data.
    # run.finish()

    df = pd.read_parquet("data/aisdk/processed/windows_30_30/cluster_id=0")
    print(df.head())

    print(df.info())

if __name__ == "__main__":
    main()
    
# TODO: do we need to filter on ships going from/to ports in dk