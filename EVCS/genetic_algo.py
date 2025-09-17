import pandas as pd
import numpy as np

# --- LOAD DATA ---
data = pd.read_csv('evcs_input_gurugram.csv')
candidate_buses = data['BusNo'].tolist()

# Example: Infrastructure cost and user cost functions
def infra_cost(landcost, num_connectors=4, Cint=300000, Ccon=50000, PC=1):
    return Cint + 25 * landcost * num_connectors + PC * Ccon * (num_connectors - 1)

def user_cost(popdensity, proximity, base_cost=100000):
    # Example user cost function (you can update with your actual model)
    return base_cost + 100 * popdensity + 2000 * (5 - proximity)  # Smaller proximity (to highway) means less extra cost

data['InfraCost'] = data['LandCost'].apply(lambda x: infra_cost(x))
data['EVUserCost'] = data.apply(lambda row: user_cost(row['PopulationDensity'], row['ProximityToHighway']), axis=1)

# --- NORMALIZE (Min-Max) ---
for col in ['InfraCost', 'EVUserCost']:
    data[col + '_norm'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# --- WEIGHTED SUM (You can adjust weights) ---
weight_infra = 0.5
weight_user = 0.5
data['WeightedSum'] = data['InfraCost_norm'] * weight_infra + data['EVUserCost_norm'] * weight_user

# --- GENETIC ALGORITHM TO FIND BEST 25 LOCATIONS ---
POP_SIZE = 30
NUM_SELECTED = 25
MAX_ITER = 50

def initialize_population(pop_size, candidates, num_selected):
    return [np.random.choice(candidates, num_selected, replace=False).tolist() for _ in range(pop_size)]

def fitness(ind, data):
    # Minimize total weighted sum
    return -data[data['BusNo'].isin(ind)]['WeightedSum'].sum()

population = initialize_population(POP_SIZE, candidate_buses, NUM_SELECTED)
best_individual = None
best_fitness = float('-inf')

for iteration in range(MAX_ITER):
    fitnesses = [fitness(ind, data) for ind in population]
    for ind, fit in zip(population, fitnesses):
        if fit > best_fitness:
            best_fitness = fit
            best_individual = ind
    # Selection
    selected = []
    for _ in range(POP_SIZE):
        i, j = np.random.randint(POP_SIZE, size=2)
        selected.append(population[i] if fitnesses[i] > fitnesses[j] else population[j])
    # Crossover and mutation
    new_population = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = selected[i], selected[min(i+1, POP_SIZE-1)]
        cut = np.random.randint(1, NUM_SELECTED-1)
        c1 = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
        c2 = p2[:cut] + [x for x in p1 if x not in p2[:cut]]
        c1 = c1[:NUM_SELECTED]
        c2 = c2[:NUM_SELECTED]
        new_population.extend([c1, c2])
    # Mutation
    for ind in new_population:
        if np.random.rand() < 0.2:
            idx = np.random.randint(NUM_SELECTED)
            possible = [x for x in candidate_buses if x not in ind]
            if possible:
                ind[idx] = np.random.choice(possible)
    population = new_population

# --- SAVE BEST INDIVIDUAL LOCATION-WISE ---
result = data[data['BusNo'].isin(best_individual)][['BusNo', 'Location', 'InfraCost', 'EVUserCost', 'WeightedSum']]
result.to_csv('evcs_best_25_locations.csv', index=False)
print("\nBest 25 EVCS locations saved to 'evcs_best_25_locations.csv'")
print(result)