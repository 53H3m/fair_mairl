import itertools
import math
from collections import defaultdict

import numpy as np
import pandas
import tree

from irl_types import Config
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()
sns.set_context("paper")
sns.set_theme(style="white", rc={
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.bottom": True, "ytick.left": True,
})

# Must plot
# relative fairness ?

gaps = ["PSG", "MNIG"]
samplings = ["ULA", "MH"]
seeds = range(10)

traj_pools = [
    (1, 3),
    (2, 3),
    (4, 3),
    (8, 3),
    (16, 3),
    (32, 3),
    (64, 3),
    (128, 3),

    (3, 2),
    (6, 2),
    (12, 2),
    (24, 2),
    (48, 2),
    (96, 2),
    (192, 2),
    (384, 2),
]

ticks = [
    n * t
    for t, n in traj_pools[:8]
]
tick_labels = [
    str(n) for n in ticks
]



configs = {
    (gap, sampling, num_traj, pool_size): Config(
        num_trajectories_per_pairing=num_traj,
        pool_size=pool_size,
        sampling_algo=sampling,
        sampling_method="PORP",
        gap_function=gap
    )
    for gap, sampling, (num_traj, pool_size) in itertools.product(gaps, samplings, traj_pools)
}
configs.update(
    {
    (sampling, num_traj, pool_size): Config(
        num_trajectories_per_pairing=num_traj,
        pool_size=pool_size,
        sampling_algo=sampling,
        sampling_method="DRP",
    )
    for gap, sampling, (num_traj, pool_size) in itertools.product(gaps, samplings, traj_pools)
    }
)

evals = {}

for i, (name, config) in enumerate(configs.items()):
    evals[name] = config.load_eval()
    print(f"{i+1}/{len(configs)}")


def eval_fairness(first_two=True):
    data = defaultdict(list)

    for name, e in evals.items():

        config = configs[name]

        run_data = e["run_data"]

        true_diffs = []

        for i, error in enumerate(e["fairness_error"]):
            # to debug plotting
            if i not in seeds:
                continue

            mean_over_samples = np.mean(error, axis=0)
            data["base_reward_error"].append(np.sum(np.mean(e["base_reward_error"][i], axis=0)))
            if first_two:
                mean_over_samples = mean_over_samples[:2]

            total_error = np.sum(mean_over_samples)
            data["seed_id"].append(i)
            data["fairness_error"].append(total_error)

            tree.map_structure_with_path(
                lambda p, v: data[p].append(v),
                config
            )

            true_fairness = run_data[i].solution.fairness_levels

            def pairwise_diff(vec):
                return np.array(vec)[:, None] - np.array(vec)[None, :]

            if first_two:
                true_diff = pairwise_diff(true_fairness[:2])

                relative_errors = [
                    np.sum(np.abs(true_diff - pairwise_diff(x.fairness_levels[:2]))) for x in
                    run_data[i].reward_samples
                ]
            else:
                true_diff = pairwise_diff(true_fairness)
                relative_errors = [
                    np.sum(np.abs(true_diff - pairwise_diff(x.fairness_levels))) for x in
                    run_data[i].reward_samples
                ]

            data["relative_fairness_error"].append(np.mean(relative_errors))

            n_comb = math.comb(config.pool_size, 2)

            data["num_traj"].append(n_comb * config.num_trajectories_per_pairing)

            data["sampler"].append(config.sampling_algo)
            data["gap_function"].append(None if config.gap_function is None else config.gap_function.replace('MNIG', 'NIG'))
            data["Ablation"].append("2 Agents" if n_comb == 1 else "3 Agents")
            data["method"].append(config.sampling_method)


            if config.sampling_method == "DRP":
                run_name = f"{config.sampling_algo}-{config.sampling_method}"
            else:
                run_name = f"{config.sampling_algo}-{config.sampling_method}-{config.gap_function.replace('MNIG', 'NIG')}"

            data["label"].append(run_name)

    palette = sns.color_palette("hls", 6)


    df = pandas.DataFrame(data)

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'wspace': 0.2})
    # --- Plot 1: Fairness Error ---
    df['fairness_error_scaled'] = df['fairness_error'] / (2 * 2 / 3)

    sns.lineplot(
        data=df,
        x="num_traj",
        y="fairness_error_scaled",
        hue="label",
        errorbar=None,  # "se",
        err_style="bars",
        linewidth=1.5,
        palette=palette,
        style="Ablation",
        ax=ax1,
        err_kws={'capsize': 2},
        markers=True,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Total Number of Trajectories", fontsize=21, labelpad=4)
    ax1.set_ylabel("Absolute Fairness Error", fontsize=21, labelpad=4)
    ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_xticks(ticks, tick_labels)
    ax3.tick_params(labelsize=15)
    ax1.legend_.remove()
    df['relative_fairness_error_scaled'] = df['relative_fairness_error'] / (2.5)

    # sns.lineplot(
    #     data=df,
    #     x="num_traj",
    #     y="relative_fairness_error_scaled",
    #     hue="label",
    #     errorbar=None,#"se",
    #     err_style="bars",
    #     linewidth=1.5,
    #     palette=palette,
    #     style="Ablation",
    #     ax=ax2,
    #     err_kws={'capsize': 2},
    #     markers=True,
    # )
    # #ax2.hlines(2.5, xmin=3, xmax=3 * 128, color="black", linestyle=":", label="Random", linewidth=1.5)
    # ax2.set_xscale("log")
    # # ax1.set_yscale("log")
    # ax2.set_xlabel("Total Number of Trajectories", fontsize=18, labelpad=6)
    # ax2.set_ylabel("Relative Fairness Error", fontsize=18, labelpad=4)
    # ax2.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    # ax2.set_xticks(ticks, tick_labels)
    # yticks = [0.1, 0.3, 0.5, 0.7, 1]
    # # ax1.set_yticks(yticks, [str(ytick) for ytick in yticks])
    # ax2.legend_.remove()

    df['base_reward_error_scaled'] = df['base_reward_error'] / (5 * 3 / 3)
    # --- Plot 2: Base Reward Error ---
    sns.lineplot(
        data=df,
        x="num_traj",
        y="base_reward_error_scaled",
        hue="label",
        errorbar=None,  #"se",
        err_style="bars",
        linewidth=1.5,
        palette=palette,
        style="Ablation",
        ax=ax3,
        err_kws={'capsize': 2.},
        markers=True,
    )
    ax3.tick_params(labelsize=15)
    ax3.set_xscale("log")
    ax3.set_xlabel("Total Number of Trajectories", fontsize=21, labelpad=4)
    ax3.set_ylabel("Absolute Intrinsic Rewards Error", fontsize=21, labelpad=6)
    ax3.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.set_xticks(ticks, tick_labels)

    ax3.legend_.remove()

    handles, labels = ax1.get_legend_handles_labels()
    handles.pop(0)
    labels.pop(0)

    handles.pop(-3)
    labels.pop(-3)

    ax1.minorticks_off()
    #ax2.minorticks_off()
    ax3.minorticks_off()

    fig.legend(handles, labels, loc='upper center', ncol=10, frameon=True, fontsize=18, labelspacing=0, columnspacing=0.8)

    #plt.tight_layout()
    plt.savefig("random_mg_plots.png")
    plt.show()


eval_fairness()
