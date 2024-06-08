from pprint import pprint
import pandas as pd
from Support.path import get_path_df
from scripts.machine_learning_algorithms.simple_linear_regression import SLR
from scripts.machine_learning_algorithms.multiple_linear_regression import MLR
from scripts.machine_learning_algorithms.simple_tree import ST
from scripts.machine_learning_algorithms.polynomial_regression import PR


def main():
    bob = pd.read_csv(get_path_df([''], 'bob.csv'))
    target, factors = ['calories'], ['distance', 'temp', 'duration']

    lr = SLR(target, factors,  bob)
    lr.print_information()

    mr = MLR(target, factors, bob)
    mr.print_information()

    # st = ST(target, factors, bob)
    # st.print_information()

    pr_range_20 = [PR(target, ['duration'], bob, i) for i in range(1, 2)]

    for degree in range(len(pr_range_20)):
        pr_range_20[degree].print_information()

    pr_5 = [PR(target, [label], bob, degree)
            for label in factors for degree in range(1, 6)]
    for mod in pr_5:
        mod.print_information()
        print()


if __name__ == '__main__':
    main()
