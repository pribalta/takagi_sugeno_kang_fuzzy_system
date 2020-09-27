import argparse
from typing import Tuple

import numpy as np

parser = argparse.ArgumentParser(description='Takagi-Sugeno fuzzy system for FEHI')

parser.add_argument('--dataset', type=str, help='Dataset to use in the experiment')


def parse_dataset(path: str) -> Tuple:
    """
    Load CSV file from storage and parse the data inside.

    :param path: (str) Path to the CSV file
    :return: (Tuple) X and y values extracted from the CSV file
    """
    with open(path, 'r') as f:
        data = f.readlines()

    clean_rows = [row.strip().split(',') for row in data]
    clean_rows = np.array([list(map(float, row)) for row in clean_rows])

    return clean_rows[:, :-1], clean_rows[:, -1]


def main():
    """Entry point of the application"""
    flags = parser.parse_args()

    x, y = parse_dataset(flags.dataset)


if __name__ == '__main__':
    main()
