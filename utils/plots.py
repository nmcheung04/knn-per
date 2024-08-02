import os
import numpy as np

import matplotlib.pyplot as plt

ALPHAS = ["0.1", "0.3", "0.5", "0.7", "1.0"]


def plot_capacity_effect(ax, results_dir, save_path=None, label=None):
    """plot the effect of the interpolation parameter on the test accuracy

    :param ax:

    :param results_dir: directory storing the experiment results

    :param save_path: directory to save the plot, default is `None`, if not provided the plot is not saved

    :param label: label of the plot, default is None

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_test_samples = np.load(os.path.join(results_dir, "n_test_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))
    capacities_grid = np.load(os.path.join(results_dir, "capacities_grid.npy"))

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_test_samples.sum()
    print(f"Average scores:  {average_scores}")

    for jj, capacity in enumerate(capacities_grid):
        mse_scores = average_scores[:, jj]
        # print(f"mse_scores: {mse_scores}")
        ax.plot(
            weights_grid,
            mse_scores,
            linewidth=5.0,
            label=label,
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax.grid(True, linewidth=2)

        ax.set_ylabel("MSE", fontsize=50)
        ax.set_xlabel("Capacity", fontsize=50)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=25)

        plt.savefig(save_path, bbox_inches='tight')


def plot_weight_effect(results_dir, save_path=None):
    """

    :param results_dir:
    :param save_path:
    :return:

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_test_samples = np.load(os.path.join(results_dir, "n_test_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_test_samples.sum()

    fig, ax = plt.subplots(figsize=(12, 10))

    mse_scores = average_scores[:, 2]
    ax.plot(
        weights_grid,
        mse_scores,
        linewidth=5.0,
        label=r"$\bar{n}_{m} = 5$"
    )

    mse_scores = average_scores[:, 10]
    ax.plot(
        weights_grid,
        mse_scores,
        linewidth=5.0,
        linestyle="dashdot",
        label=r"$\bar{n}_{m} = 25$"
    )

    mse_scores = average_scores[:, 25]
    ax.plot(
        weights_grid,
        mse_scores,
        linewidth=5.0,
        linestyle="dashed",
        label=r"$\bar{n}_{m} = 100$"
    )

    mse_scores = average_scores[:, -1]
    ax.plot(
        weights_grid,
        mse_scores,
        linewidth=5.0,
        linestyle="dotted",
        label=r"$\bar{n}_{m} = 250$"
    )

    ax.grid(True, linewidth=2)

    ax.set_ylabel("MSE", fontsize=50)
    ax.set_xlabel(r"$\lambda$", fontsize=50)
    ax.tick_params(axis='both', labelsize=25)

    ax.legend(fontsize=25)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_hetero_effect(results_dir, save_path=None):
    _, ax = plt.subplots(figsize=(12, 10))

    for alpha in ALPHAS:

        current_dir = os.path.join(results_dir, f"n_neighbors_7_alpha_{alpha}")
        label = r"$\alpha$={}".format(alpha)

        plot_capacity_effect(ax, results_dir=current_dir, label=label)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax.grid(True, linewidth=2)

        ax.set_ylabel("MSE", fontsize=50)
        ax.set_xlabel("Capacity", fontsize=50)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=25)

        plt.savefig(save_path, bbox_inches='tight')

    else:
        plt.show()

    plt.close()
