from itertools import combinations


def get_all_combinations(collection: list[str]):
    return [subset for i in range(1, len(collection) + 1) for subset in combinations(collection, i)]
