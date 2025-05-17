from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_theme()
sns.set_context("paper")
sns.set_style("ticks")



data_files = {
    r"$2 \times 2$": {
        r"$3$ Trajectories" : "Zs_gapv_1_2_2_3.npy",
        # r"$18$ Trajectories" : "Zs_seed_1_dim_2x2_traj_6.npy",
        r"$36$ Trajectories": "Zs_gapv_1_2_2_36.npy",
    },
    r"$3 \times 3$": {
    r"$3$ Trajectories": "Zs_gapv_0_3_3_3.npy",
    #r"$18$ Trajectories" : "Zs_seed_0_dim_3x3_traj_6.npy",
    r"$36$ Trajectories": "Zs_gapv_0_3_3_36.npy",},
     r"$4 \times 4$": {
         r"$3$ Trajectories" : "Zs_gapv_2_4_4_3.npy",
         #r"$18$ Trajectories" : "Zs_seed_2_dim_4x4_traj_6.npy",
         r"$36$ Trajectories": "Zs_gapv_2_4_4_36.npy",
     },
    r"$5 \times 5$": {
        r"$3$ Trajectories" : "Zs_gapv_1_5_5_3.npy",
        # r"$18$ Trajectories" : "Zs_seed_1_dim_2x2_traj_6.npy",
        r"$36$ Trajectories": "Zs_gapv_1_5_5_36.npy",
    },
}

dd = {
    "Z": [],
    "label" : [],
    "runtype": []
}

for label, runs in data_files.items():
    for runtype, filename in runs.items():
        with open(filename, "rb") as f:
            data = np.load(f)

        estimated_zs = data[-1] / len(data)

        print(np.min(estimated_zs) / np.max(estimated_zs))

        estimated_zs /= np.max(estimated_zs)

        dd["Z"].extend(list(estimated_zs))
        dd["label"].extend([label] * len(estimated_zs))
        dd["runtype"].extend([runtype] * len(estimated_zs))

df = pd.DataFrame(dd)

# === Plot ===
colors = ["#A3BCE2", "#F2B89C"]


sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(7, 4))

ax = sns.violinplot(y="Z", x="label", data=df, cut=0, hue="runtype", palette=colors,
                    scale="count", inner="box", bw=.25, linewidth=0.5,
                    inner_kws={"color": "black"})

# Labels and aesthetics
ax.set_xlabel(r"")
ax.set_ylabel(r"Rescaled $Z$", fontsize=12)
ax.set_ylim(0.34, 1)
ax.set_xticks(list(data_files.keys()))
ax.set_yticks([1., 0.8, 0.6, 0.4])
ax.set_yticklabels(["1", "0.8", "0.6", "0.4"])
ax.minorticks_off()

legend_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(colors, data_files[r"$3 \times 3$"].keys())
]

ax.legend(handles=legend_handles, loc="lower left", frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig("test")
plt.show()

