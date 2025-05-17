import itertools
import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

posteriors = [
    "PORP",
    "DRP"
]
gaps = [
    "PSG", "NIG",
]
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

commands = list({
    f"python3 random_mg.py -p {posterior} -g {gap} -s {sampling} --seed {seed} --pool_size {pool_size} --num_traj {num_traj}"
    for posterior, gap, sampling, seed, (num_traj, pool_size) in itertools.product(posteriors, gaps, samplings, seeds, traj_pools)
})
np.random.shuffle(commands)
commands = []

eval_commands = list({
    f"python3 random_mg.py -e -p {posterior} -g {gap} -s {sampling} -n {len(seeds)} --pool_size {pool_size} --num_traj {num_traj}"
    for posterior, gap, sampling, (num_traj, pool_size) in itertools.product(posteriors, gaps, samplings, traj_pools)
})

print(f"Running the following {len(commands)} commands:")
for command in commands:
    print(f" >> {command}")
print()

def run_command(cmd):
    print(f"Running: {cmd}")
    x = subprocess.run(cmd, shell=True,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL
                          )
    print(f"Job done: {cmd}")
    return x

if __name__ == "__main__":
    with Pool(processes=os.cpu_count()//4 + 2, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap_unordered(run_command, commands), total=len(commands)):
            pass
        print('Running eval.')
        for _ in tqdm(pool.imap_unordered(run_command, eval_commands), total=len(eval_commands)):
            pass
