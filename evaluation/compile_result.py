import pandas as pd

import numpy as np

import argparse

def main(args):
    data = pd.read_csv(args.result_file)

    mean = data.mean()
    CI = data.std() * 1.96 / np.sqrt(len(data))

    compiled_result = (pd.concat([mean, CI], axis=1))
    compiled_result.columns = ['Mean', '95CI']
    print(compiled_result)
    compiled_result.to_csv(args.result_file[:-4] + '_compiled.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct the mean and 95 CI")
    parser.add_argument('--result_file', type=str, help='result file')
    args = parser.parse_args()
    main(args)    