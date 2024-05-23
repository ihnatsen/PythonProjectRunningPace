import pandas as pd
from Support.path import get_path_df
from scripts.machine_learning_algorithms.regression import LinearRegression


def main():
    bob = pd.read_csv(get_path_df([''], 'bob.csv'))
    factors, target = ['distance', 'temp', 'pressure', 'duration', 'avg_pace'], ['calories']
    lr = LinearRegression(bob, factors, target)
    lr.print_information()


if __name__ == '__main__':
    main()
