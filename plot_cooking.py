import itertools
import math
from collections import defaultdict

import numpy as np
import pandas
import tree

from irl_types import Config
import matplotlib.pyplot as plt
import seaborn as sns

palette = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9"]
sns.set_context("paper")
sns.set_theme(style="white", rc={
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.bottom": True, "ytick.left": True,
})

posteriors = ["PORP"]
gaps = ["PSG"]
samplings = ["ULA"]
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
    (256, 3),
    (512, 3),
    (1024, 3),
]


ticks = [
    n * t
    for t, n in traj_pools
]#[::5]
tick_labels = [str(t) for t in ticks]

# tick_labels = []
# for tick in ticks:
#     sci_notation = f"{tick:.1e}"
#     base, exponent = sci_notation.split('e')
#     exponent = int(exponent)  # Convert exponent to integer
#     # Format in LaTeX-style
#     formatted_number = f"${int(float(base))} \\times 10^{{{exponent}}}$"
#
#     tick_labels.append(formatted_number)


configs = {
    (gap, sampling, num_traj, pool_size): Config(
        num_trajectories_per_pairing=num_traj,
        pool_size=pool_size,
        sampling_algo=sampling,
        sampling_method="PORP",
        gap_function=gap,
        exp_type="cooking"
    )
    for gap, sampling, (num_traj, pool_size) in itertools.product(gaps, samplings, traj_pools)
}


evals = {}

for i, (name, config) in enumerate(configs.items()):
    evals[name] = config.load_eval()
    print(f"{i+1}/{len(configs)}")


# 1. Eval fairness precision
# x is logscale of num of samples
# y is absolute error, meaned over sampled items
# could also do relative error (less strict and still interesting)

states_visited = [0, 1, 2, 3, 4, 5, 7, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 69, 70, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 100, 101, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 121, 122, 123, 124, 125, 127, 129, 130, 131, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 151, 152, 153, 154, 155, 157, 159, 160, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 187, 189, 190, 192, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 207, 208, 209, 211, 212, 213, 214, 215, 217, 219, 220, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 239, 241, 242, 243, 244, 245, 247, 249, 250, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 271, 272, 273, 275, 277, 279, 280, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 301, 302, 303, 304, 305, 307, 309, 310, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 334, 335, 339, 340, 341, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358]
not_visited = [i for i in range(390) if i not in states_visited]
goal_states = [192, 193, 194, 195, 207, 208, 209, 210, 222, 223, 224, 225, 237, 238, 239, 240, 252, 253, 254, 255, 267, 268, 269, 270]
goals_visited = [g for g in goal_states if g in states_visited]
other_states = [s for s in range(390) if s not in goal_states]
other_visited_states = [s for s in states_visited if s not in goals_visited]
print(len(states_visited))


def eval_fairness(first_two=True):
    data = defaultdict(list)

    fairness_posteriors = defaultdict(list)



    for name, e in evals.items():

        config = configs[name]
        run_data = e["run_data"]


        for i, error in enumerate(e["fairness_error"]):
            # to debug plotting
            if i not in seeds:
                continue

            mean_over_samples = np.mean(error, axis=0)
            data["Other States (Visited)"].append(2 * np.mean(np.mean(e["base_reward_error"][i], axis=0)[other_visited_states]))
            data["Delivery States (Visited)"].append(2 * np.mean(np.mean(e["base_reward_error"][i], axis=0)[goals_visited]))
            data["Other States"].append(2 * np.mean(np.mean(e["base_reward_error"][i], axis=0)[other_states]))
            data["Delivery States"].append(2 * np.mean(np.mean(e["base_reward_error"][i], axis=0)[goal_states]))


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
                    np.linalg.norm(true_diff - pairwise_diff(x.fairness_levels[:2]), ord=1) for x in
                    run_data[i].reward_samples
                ]
            else:
                true_diff = pairwise_diff(true_fairness)
                relative_errors = [
                    np.linalg.norm(true_diff - pairwise_diff(x.fairness_levels), ord=1) for x in
                    run_data[i].reward_samples
                ]

            data["relative_fairness_error"].append(np.mean(relative_errors))

            n_comb = math.comb(config.pool_size, 2)
            num_trajectories = n_comb * config.num_trajectories_per_pairing
            data["num_traj"].append(num_trajectories)

            run_name = f"{config.sampling_algo}-{config.sampling_method}-{config.gap_function}"

            data["label"].append(run_name)

            fairness_samples = [x.fairness_levels for x in run_data[i].reward_samples]

            for k, agent in enumerate([f"Agent {n}" for n in [1, 3, 2]]):
                for f in fairness_samples:
                    fairness_posteriors["Agent"].append(agent)
                    fairness_posteriors["Fairness Level"].append(f[k])
                    fairness_posteriors["Seed"].append(i)
                    fairness_posteriors["Number of Trajectories"].append(num_trajectories)


            # if config.gap_function == "MNIG":
            #     print(run_name)
            #     print(run_data[i].reward_samples[0].reward_func)
            #     input()

    df = pandas.DataFrame(data)

    dist_pd = pandas.DataFrame(fairness_posteriors)

    values = [0, 0.5, 1]
    colors = ["b", "salmon", "seagreen"]

    palettes = [sns.light_palette(color, n_colors=len(seeds)+5)[1:-4] for color in colors]
    dark_palettes = [sns.dark_palette(color, n_colors=len(seeds)) for color in colors]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    ss = [1, 1, 1]

    for idx, (n, t) in enumerate([traj_pools[4], traj_pools[7], traj_pools[10]]):
        ax = axes[idx]
        plt.sca(ax)

        subset = dist_pd[dist_pd["Number of Trajectories"] == n * t]

        # Plot True Fairness line
        plt.axvline(x=0, color="#000000ff", linestyle='--', linewidth=2, label="True Fairness" if n*t==16*3 else None)

        # Loop through agents and plot KDEs
        for i, agent in enumerate([f"Agent {n}" for n in [1, 2, 3]]):
            filtered = subset[(subset["Seed"] == 1) & (subset["Agent"] == agent)]

            sns.kdeplot(data=filtered, x="Fairness Level",
                        common_norm=False, fill=True,
                        color=colors[i], alpha=0.7,
                        label=None if n*t != 3*16 else agent,
                        linewidth=2, clip=(0., 1.), bw_adjust=0.2, ax=ax)

            xticks = [0, 0.5, 1]
            ax.set_xticks(xticks, [str(xtick) for xtick in xticks])
            ax.set_ylim(0, 25)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=18)

            plt.axvline(x=values[i], color=colors[i], linestyle='--', linewidth=2)

        ax.set_title(f"{n*t} Trajectories", fontsize=20)

        #ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    #fig.supxlabel("Fairness Level", fontsize=26)
    #fig.supylabel("Density", fontsize=26)

    fig.legend(loc='lower center', ncol=10, frameon=True, fontsize=28, labelspacing=0, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    #plt.tight_layout()
    plt.savefig("fairness_posterior.png")
    plt.show()


    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    melted_df = pandas.melt(
        df,
        id_vars="num_traj",
        value_vars=["Delivery States", "Delivery States (Visited)", "Other States", "Other States (Visited)"],
        var_name="Type",
        value_name="Rate"
    )

    # Create new columns for Category and Style
    melted_df["$(s,a)$"] = melted_df["Type"].apply(lambda x: "Others" if "Other" in x else "Deliveries")
    melted_df["Style"] = melted_df["Type"].apply(lambda x: "Visited" if "Visited" in x else "All")

    # Plot with hue and style
    ax = sns.lineplot(
        data=melted_df,
        x="num_traj",
        y="Rate",
        hue="$(s,a)$",
        style="Style",
        errorbar="se",
        err_style="bars",
        markers=True,
        linewidth=2,
        palette=palette,
        err_kws={'capsize': 2},
        ax=ax,
    )

    # Log scale for X-axis
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.set_ylim(-0.05, 1.6)
    ax.tick_params(labelsize=18)

    # Axis labels
    #ax.set_xlabel("", fontsize=26, labelpad=6)
    ax.set_xlabel("Total Number of Trajectories", fontsize=26, labelpad=6)
    ax.set_ylabel("Absolute Intrinsic Rewards Error", fontsize=22, labelpad=6)

    handles, labels = ax.get_legend_handles_labels()
    ax.minorticks_off()

    ax.legend_.remove()

    handles.pop(0)
    labels.pop(0)
    handles.pop(-3)
    labels.pop(-3)

    #fig.legend(handles, labels, loc='lower center', ncol=10, frameon=True, fontsize=22, labelspacing=0, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))

    # Tight layout and grid customization
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig("cooking_rewards.png")
    plt.show()


eval_fairness(first_two=False)
