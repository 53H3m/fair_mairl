# Inference of Intrinsic Rewards and Fairness in Multi-Agent Systems

This repository contains the source code for the experiments and analysis presented in the study on *Inference of Intrinsic Rewards and Fairness in Multi-Agent Systems*. The experiments cover two main settings: **Randomised Markov Game** and **Collaborative Cooking Game**.

---

## **Setup**

* **Tested on:** Ubuntu 22.04, Python 3.11
* **Dependencies:** Install all required packages with:

  ```bash
  pip install -r requirements.txt
  ```

---

## **Randomised Markov Game Experiments**

You can run experiments for the Randomised Markov Game environment using the following command:

```bash
python random_mg.py -p {DRP|PORP} -g {NIG|PSG} -s {ULA|MH} --seed {seed} --pool_size {2|3} --num_traj {num_traj}
```

### **Batch Evaluation**

To evaluate a set of runs with seeds `[0, 1, ..., n-1]`, use:

```bash
python random_mg.py -e -p {DRP|PORP} -g {NIG|PSG} -s {ULA|MH} -n {n} --pool_size {2|3} --num_traj {num_traj}
```

### **Data Storage & Plotting**

* All run data is stored in `data/random_mg`.
* To generate plots for analysis, run:

  ```bash
  python plot_random_mg.py
  ```

---

## **Collaborative Cooking Game Experiments**

First, generate the Markov game environment and equilibria by running:

```bash
python -m games.cooking_mg
```

This will also produce GIFs visualizing the policies for each equilibrium.

### **Running Experiments**

To execute experiments in this environment, use:

```bash
python cooking.py -p {DRP|PORP} -g {NIG|PSG} -s {ULA|MH} --seed {seed} --pool_size {2|3} --num_traj {num_traj}
```

> **Note:** These experiments are memory-intensive, particularly for `PORP-NIG` and `DRP`, as TensorFlow is used for computation.

### **Batch Evaluation**

To evaluate multiple runs with seeds `[0, 1, ..., n-1]`, use:

```bash
python cooking.py -e -p {DRP|PORP} -g {NIG|PSG} -s {ULA|MH} -n {n} --pool_size {2|3} --num_traj {num_traj}
```

### **Data Storage & Plotting**

* All run data is stored in `data/cooking`.
* To generate plots for analysis, run:

  ```bash
  python plot_cooking.py
  ```

---

## **Hyperparameters**

Key hyperparameters for the experiments can be found in:

* `cooking.py` for Collaborative Cooking experiments
* `random_mg.py` for Randomised Markov Game experiments

---

## Normalising constant experiments

To evaluate the constant nature of the normalising constant, over policy deployments, use:

  ```bash
  python normalising_constant_tests.py
  ```

---