from pprint import pprint
import pandas as pd
from Support.path import get_path_df
from scripts.machine_learning_algorithms.simple_linear_regression import SLR
from scripts.machine_learning_algorithms.multiple_linear_regression import MLR
from scripts.machine_learning_algorithms.simple_tree import ST


def main():
    bob = pd.read_csv(get_path_df([''], 'bob.csv'))
    target, factors = ['calories'], ['temp', 'pressure', 'duration', 'distance']

    lr = SLR(target, factors,  bob)
    lr.print_information()

    mr = MLR(target, factors, bob)
    mr.print_information()

    # st = ST(target, factors, bob)
    # st.print_information()


if __name__ == '__main__':
    main()
