import itertools
import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

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

commands = list({
    f"python3 cooking.py -p {posterior} -g {gap} -s {sampling} --seed {seed} --pool_size {pool_size} --num_traj {num_traj}"
    for posterior, gap, sampling, seed, (num_traj, pool_size) in itertools.product(posteriors, gaps, samplings, seeds, traj_pools)
})

eval_commands = list({
    f"python3 cooking.py -e -p {posterior} -g {gap} -s {sampling} -n {len(seeds)} --pool_size {pool_size} --num_traj {num_traj}"
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

    for cmd in tqdm(commands):
        run_command(cmd)
    for cmd in tqdm(eval_commands):
        run_command(cmd)
