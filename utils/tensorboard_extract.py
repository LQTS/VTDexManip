import os

from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm


def tensorboard2csv(in_path, ex_path):
    # load log data

    event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        # print(key)
        if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
    os.makedirs(os.path.dirname(ex_path), exist_ok=True)
    df.to_csv(ex_path)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex-path', type=str, required=True, help='location to save the exported data')

    args = parser.parse_args()
    tensorboard2csv(args.in_path, args.ex_path)

