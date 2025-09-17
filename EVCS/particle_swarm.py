import pandas as pd
import numpy as np

# ---- Load Input Data ----
data = pd.read_csv('evcs_input_50_locations.csv')
N = len(data)
NUM_SELECT = 25   # Number of optimal locations you want to find (change as needed)
POP_SIZE = 40
MAX_ITER = 60

# ---- Annual Electricity Cost function ----
def annual_electricity_cost(row):
    num_days = 365
    evs_per_day = row['AvgEVPerDay']
    soc_arr = row['AvgSOC_Arrival']
    soc_dep = row['AvgSOC_Departure']
    batt = row['AvgBattery_kWh']
    eff = row['ChargingEfficiency']
    ecost = row['ElectricityCost']
    if evs_per_day == 0:
        return 0
    session_energy = (soc_dep - soc_arr) * batt / eff
    return num_days * evs_per_day * session_energy * ecost

data['AnnualElecCost'] = data.apply(annual_electricity_cost, axis=1)

# ---- Normalize Features for Weighted Sum ----
for col in ['AnnualElecCost', 'LandCost']:
    data[col+'_norm'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# ---- Define Weighted Sum Fitness Function ----
def weighted_sum(indices, data, weights):
    subset = data.iloc[indices]
    s = 0
    for col, w in weights.items():
        s += w * subset[col].sum()
    return s

weights = {'AnnualElecCost_norm': 0.6, 'LandCost_norm': 0.4}  # Adjust as needed

# ---- Particle Swarm Optimization ----
class Particle:
    def __init__(self, num_select, N):
        self.position = np.random.choice(range(N), num_select, replace=False)
        self.velocity = np.zeros(num_select)
        self.best_position = self.position.copy()
        self.best_score = np.inf

def pso_optimize(data, weights, num_select, pop_size, max_iter):
    N = len(data)
    swarm = [Particle(num_select, N) for _ in range(pop_size)]
    global_best_pos = None
    global_best_score = np.inf

    for iteration in range(max_iter):
        for particle in swarm:
            score = weighted_sum(particle.position, data, weights)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_pos = particle.position.copy()
        # PSO velocity/position update (random re-sampling for combinatorial case)
        for particle in swarm:
            # Randomly swap one or more positions with personal or global best
            if np.random.rand() < 0.6:
                # Move toward personal best
                swap_idx = np.random.randint(num_select)
                if particle.position[swap_idx] not in particle.best_position:
                    choices = [i for i in particle.best_position if i not in particle.position]
                    if choices:
                        particle.position[swap_idx] = np.random.choice(choices)
            if np.random.rand() < 0.3:
                # Move toward global best
                swap_idx = np.random.randint(num_select)
                if particle.position[swap_idx] not in global_best_pos:
                    choices = [i for i in global_best_pos if i not in particle.position]
                    if choices:
                        particle.position[swap_idx] = np.random.choice(choices)
            # Random mutation (exploration)
            if np.random.rand() < 0.1:
                swap_idx = np.random.randint(num_select)
                choices = [i for i in range(N) if i not in particle.position]
                if choices:
                    particle.position[swap_idx] = np.random.choice(choices)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Current best score = {global_best_score:.4f}")
    return global_best_pos, global_best_score

# ---- Run PSO ----
best_indices, best_score = pso_optimize(data, weights, NUM_SELECT, POP_SIZE, MAX_ITER)
best_locs = data.iloc[best_indices].copy()
print("\nBest selected EVCS locations:")
print(best_locs[['BusNo', 'Location', 'AnnualElecCost', 'LandCost']])

# ---- Save results as CSV for next code ----
best_locs.to_csv('evcs_pso_selected_locations.csv', index=False)
print("\nSaved best EVCS locations to 'evcs_pso_selected_locations.csv'")
