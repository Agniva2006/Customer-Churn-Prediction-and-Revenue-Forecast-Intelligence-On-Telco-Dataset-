import matplotlib.pyplot as plt


def plot_feature_importance(series):
    series.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top Features")
    plt.show()