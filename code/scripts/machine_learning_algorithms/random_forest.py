from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scripts.machine_learning_algorithms.ML import Algorithm
from sklearn.tree import plot_tree


class RF(Algorithm):

    def create_model(self):
        mod = RandomForestRegressor(max_depth=5, n_estimators=100)
        mod.fit(self.train[self.factors],
                self.train[self.target].to_numpy().ravel())
        return mod

    def print_information(self):
        super().print_information()
        plt.figure(figsize=(20, 10))
        plot_tree(self.model.estimators_[0],
                  filled=True,
                  feature_names=self.factors,
                  fontsize=5,
                  rounded=True,
                  impurity=True,
                  max_depth=3
                  )

        plt.title("Sample Tree from the Random Forest")
        plt.show()
