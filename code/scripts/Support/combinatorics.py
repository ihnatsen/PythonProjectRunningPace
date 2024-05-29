from itertools import combinations


def get_all_combinations(collection: list[str]):
    subsets = []

    for i in range(1, len(collection)+1):
        for subset in combinations(collection, i):
            subsets.append(subset)
    return subsets

