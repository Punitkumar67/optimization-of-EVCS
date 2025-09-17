import numpy as np
import pandas as pd

np.random.seed(42)  # For reproducibility

N = 50
locations = [
    "Substation-1","DLF Phase 1","DLF Phase 2","DLF Phase 3","DLF Phase 4","DLF Phase 5",
    "Sector 14","Sector 15","Sector 21","Sector 22","Sector 23","Sector 28","Sector 29",
    "Sector 31","Sector 38","Sector 43","Sector 44","Sector 45","Sector 46","Sector 47",
    "Sector 48","Sector 49","Sector 50","Sector 51","Sector 52","Sector 53","Sector 54",
    "Sector 55","Sector 56","Sector 57","Sector 58","Sector 59","Sector 60","Sector 61",
    "Sector 62","Sector 63","Sector 65","Sector 66","Sector 67","MG Road","Cyber City",
    "Udyog Vihar","Hero Honda Chowk","IMT Manesar","Sohna Road","Palam Vihar",
    "South City 1","South City 2","Suncity","Golf Course Road"
]

# Generate random upper triangle (distances between 0.5 and 18 km)
upper = np.triu(np.random.uniform(0.5, 18, (N, N)), 1)
# Make it symmetric and set diagonal to zero
dist_matrix = upper + upper.T
np.fill_diagonal(dist_matrix, 0)

# Save as DataFrame for CSV export
df_dist = pd.DataFrame(dist_matrix, columns=locations, index=locations)
df_dist.to_csv("./data/gurugram_50loc_distance_matrix.csv")
