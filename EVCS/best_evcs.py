import pandas as pd
import numpy as np

# Parameters
NUM_SELECT = 10       # Number of locations to select
POP_SIZE = 30         # Number of wolves
MAX_ITER = 80         # Number of iterations

# Read top 25 locations
final_25 = pd.read_csv('./data/final_best_25_evcs_locations.csv')
loc_names = final_25['Location'].tolist()

# Read distance matrix (50x50), keep only the relevant 25 locations
dist_matrix_full = pd.read_csv('./data/gurugram_50loc_distance_matrix.csv', index_col=0)
dist_matrix = dist_matrix_full.loc[loc_names, loc_names].values  # 25x25 numpy array

N = len(loc_names)  # Should be 25
def fitness(selected_idx, dist_matrix):
    # selected_idx: array of selected indices (length NUM_SELECT)
    sub_matrix = dist_matrix[np.ix_(selected_idx, selected_idx)]
    # Take upper triangle without diagonal
    triu = sub_matrix[np.triu_indices(NUM_SELECT, k=1)]
    if len(triu) == 0:
        return 0
    return np.min(triu)  # maximize this!
def gwo_select(dist_matrix, N, num_select, pop_size, max_iter):
    # Initialize wolves (randomly select sets of NUM_SELECT indices)
    wolves = [np.random.choice(N, num_select, replace=False) for _ in range(pop_size)]
    wolves = [np.sort(w) for w in wolves]

    alpha, beta, delta = None, None, None
    alpha_score, beta_score, delta_score = -np.inf, -np.inf, -np.inf

    for it in range(max_iter):
        scores = []
        for wolf in wolves:
            score = fitness(wolf, dist_matrix)
            scores.append(score)
            if score > alpha_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = alpha, alpha_score
                alpha, alpha_score = wolf.copy(), score
            elif score > beta_score:
                delta, delta_score = beta, beta_score
                beta, beta_score = wolf.copy(), score
            elif score > delta_score:
                delta, delta_score = wolf.copy(), score

        a = 2 - it * (2 / max_iter)
        new_wolves = []
        for wolf in wolves:
            new_wolf = wolf.copy()
            for i in range(num_select):
                for lead in [alpha, beta, delta]:
                    if lead is not None and np.random.rand() < 0.3:
                        # Replace i-th position with corresponding position from a leader, or with a random unique index
                        if lead[i] not in new_wolf:
                            new_wolf[i] = lead[i]
                        else:
                            possible = [idx for idx in range(N) if idx not in new_wolf]
                            if possible:
                                new_wolf[i] = np.random.choice(possible)
            # Random mutation
            if np.random.rand() < 0.2:
                idx = np.random.randint(num_select)
                possible = [i for i in range(N) if i not in new_wolf]
                if possible:
                    new_wolf[idx] = np.random.choice(possible)
            new_wolves.append(np.sort(new_wolf))
        wolves = new_wolves
    return alpha, alpha_score

best_idx, best_score = gwo_select(dist_matrix, N, NUM_SELECT, POP_SIZE, MAX_ITER)

selected_locs = [loc_names[i] for i in best_idx]
final_selected = final_25.iloc[best_idx]
# add latitude and longitude from the locations_with_coordinates.csv
locations_with_coords = pd.read_csv('./data/locations_with_coordinates.csv')
locations_with_coords = locations_with_coords[['Location', 'Latitude', 'Longitude']]
final_selected = final_selected.merge(locations_with_coords, on='Location', how='left')


final_selected.to_csv('./data/final_maxdispersion_10_evcs.csv', index=False)