from itertools import combinations
from pprint import pprint

import matplotlib.pyplot as plt

from Support.path import get_path_df
from scripts.machine_learning_algorithms.simple_linear_regression import SLR
from scripts.machine_learning_algorithms.multiple_linear_regression import MLR
from scripts.machine_learning_algorithms.simple_tree import ST
from scripts.machine_learning_algorithms.polynomial_regression import PR
from scripts.machine_learning_algorithms.random_forest import RF
from scripts.machine_learning_algorithms.ML import *
from scripts.Support.combinatorics import get_all_combinations


def main():
    bob = pd.read_csv(get_path_df([''], 'bob.csv'))
    mm_data = MMScaler.set_scaler_df(bob)
    ss_data = SScaler.set_scaler_df(bob)
    target, factors = ['calories'], ['steps', 'duration', 'avg_pace', 'distance', 'temp']

    lrs = [SLR(target, [factor], bob) for factor in factors]
    for lr in lrs:
        lr.print_information()
        print()

    mrs = [MLR(target, list(combination), bob) for combination in get_all_combinations(factors)]
    for mr in mrs:
        mr.print_information()
        print()

    st = ST(target, factors, bob, 4)
    st.print_information()

    pr_5 = [PR(target, [label], bob, degree)
            for label in factors for degree in range(1, 5)]

    for mod in pr_5:
        mod.print_information()
        print()

    rf = RF(target, factors,  bob)
    rf.print_information()

    reduction = ['stp', 'dur', 'pace', 'dist', 'tmp']
    models: dict[str, Algorithm] = dict(
        [*zip([f'MLR[{','.join(combination)}]' for combination in get_all_combinations(reduction)],
              [MLR(target, list(combination), bob) for combination in get_all_combinations(factors)])]
    )
    models.update(dict(
        [*zip([f'LR[{factor}]' for factor in reduction],
              lrs)]
    ))

    models.update({'ST': st, 'RF': rf})


    r2_score = dict(sorted({name: float(model.get_r2_score()) if float(model.get_r2_score()) >= 0 else 0
                for name, model in models.items() if float(model.get_r2_score()) >= 0.5}.items(),
                    key=lambda v: v[1],
                    reverse=False))

    rmse = dict(sorted({name: float(model.get_RMSE()) if float(model.get_RMSE()) >= 0 else 0
                    for name, model in models.items() if float(model.get_RMSE()) <= 100}.items(),
                    key=lambda v: v[1],
                    reverse=True))


if __name__ == '__main__':
    main()
