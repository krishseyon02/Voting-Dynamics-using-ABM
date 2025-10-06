import matplotlib.pyplot as plt

def plot_avg(avg, title="Average opinion over time", save_path=None):
    plt.figure()
    plt.plot(range(len(avg)), avg)
    plt.xlabel("Step"); plt.ylabel("Average opinion (share of 1s)")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
