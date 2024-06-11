import pandas as pd
from Support.path import get_path_df
from scripts.machine_learning_algorithms.simple_linear_regression import SLR
from scripts.machine_learning_algorithms.multiple_linear_regression import MLR
from scripts.machine_learning_algorithms.simple_tree import ST
from scripts.machine_learning_algorithms.polynomial_regression import PR
from scripts.machine_learning_algorithms.ML import MMScaler, SScaler


def main():
    bob = pd.read_csv(get_path_df([''], 'bob.csv'))
    mm_data = MMScaler.set_scaler_df(bob)
    ss_data = SScaler.set_scaler_df(bob)
    target, factors = ['calories'], ['distance', 'duration', 'steps']

    # lr = SLR(target, factors,  bob)
    # lr.print_information()
    #
    # lr = SLR(target, factors, mm_data)
    # lr.print_information()
    #
    # mr = MLR(target, factors, bob)
    # mr.print_information()
    #
    # mr = MLR(target, factors, mm_data)
    # mr.print_information()

    st = ST(target, factors, bob)
    st.print_information()

    # pr_5 = [PR(target, [label], bob, degree)
    #         for label in factors for degree in range(1, 6)]
    #
    # for mod in pr_5:
    #     mod.print_information()
    #     print()


if __name__ == '__main__':
    main()
