"""
Process solar energy dataset, and splits it among clients
"""
import os
import re
import pandas as pd
import time
import random
import argparse

RAW_DATA_PATH = "raw_data/by_station"
TARGET_PATH = "all_data/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--s_frac',
        help='fraction of data to be used; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--train_tasks_frac',
        help='fraction of tasks / clients participating in the training; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345,
        required=False
    )

    return parser.parse_args()

def train_test_split(df, frac):
    assert 0 < frac < 1, "`frac` should be in (0, 1)"
    n = len(df)
    n_train = int(frac * n)
    train_df = df[:n_train]
    test_df = df[n_train:]
    return train_df, test_df

def save_task(dir_path, train_text, test_text):
    with open(os.path.join(dir_path, "train.csv"), 'w') as f:
        train_text.to_csv(f, index=False)
    with open(os.path.join(dir_path, "test.csv"), 'w') as f:
        test_text.to_csv(f, index=False)

def main():
    args = parse_args()

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    file_names_list = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
    n_tasks = int(len(file_names_list) * args.s_frac)
    rng.shuffle(file_names_list)
    file_names_list = file_names_list[:n_tasks]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, "test"), exist_ok=True)

    for idx, file_name in enumerate(file_names_list):
        if idx < int(args.train_tasks_frac * n_tasks):
            mode = "train"
        else:
            mode = "test"

        file_path = os.path.join(RAW_DATA_PATH, file_name)
        df = pd.read_csv(file_path)
        train_df, test_df = train_test_split(df, args.tr_frac)
        
        save_path = os.path.join(TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_text=train_df,
            test_text=test_df
        )
    print("Data processing completed.")

if __name__ == "__main__":
    main()
