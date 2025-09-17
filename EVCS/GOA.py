import pandas as pd
import numpy as np

def calc_PDGI(PDG_list, PL_list):
    """
    PDG_list: List of all DG capacities [PDG1, PDG2, ...]
    PL_list: List of all load values [PL1, PL2, ...]
    """
    PDGI = sum(PDG_list) / sum(PL_list)
    return PDGI

def mu_PDGI(PDGI, PDGI_min=0.4, PDGI_SP=0.5, PDGI_max=0.6):
    if PDGI <= PDGI_min:
        return 0
    elif PDGI_min < PDGI <= PDGI_SP:
        return (PDGI - PDGI_min) / (PDGI_SP - PDGI_min)
    elif PDGI_SP < PDGI <= PDGI_max:
        return (PDGI_max - PDGI) / (PDGI_max - PDGI_SP)
    else:
        return 0

def calc_PF(SkW, SkVA):
    # PF = cos(SkW / SkVA), but in most systems PF = SkW / SkVA
    return SkW / SkVA

def mu_PF(PF, PF_min=0.85, PF_D=0.95, PF_max=1.0):
    if PF <= PF_min:
        return 0
    elif PF_min < PF <= PF_D:
        return (PF - PF_min) / (PF_D - PF_min)
    elif PF_D < PF <= PF_max:
        return (PF_max - PF) / (PF_max - PF_D)
    else:
        return 0

def calc_APLI(Apl_DGSC, Apl_Base):
    # APL is the ratio of losses with DG+SC to base case losses
    return Apl_DGSC / Apl_Base

def mu_APLI(APLI, APLI_min=0.5, APLI_max=1.0):
    if APLI <= APLI_min:
        return 1
    elif APLI_min < APLI <= APLI_max:
        return (APLI_max - APLI) / (APLI_max - APLI_min)
    else:
        return 0

def mu_Vi(Vi, VL1=0.94, Vmin=0.95, Vmax=1.05, VL2=1.06):
    if Vi <= VL1:
        return 0
    elif VL1 < Vi < Vmin:
        return (Vi - VL1) / (Vmin - VL1)
    elif Vmin <= Vi <= Vmax:
        return 1
    elif Vmax < Vi < VL2:
        return (VL2 - Vi) / (VL2 - Vmax)
    else:
        return 0

def mu_V(V_list, VL1=0.94, Vmin=0.95, Vmax=1.05, VL2=1.06):
    # V_list: List of all bus voltages in p.u.
    # Minimum value across all buses is taken as the network's voltage fuzzy value
    return min([mu_Vi(Vi, VL1, Vmin, Vmax, VL2) for Vi in V_list])

def LPi(Ri, Pi, Qi, Vi):
    # Ri: Resistance of branch i
    # Pi, Qi: Real and reactive power at end of branch i+1
    # Vi: Voltage magnitude at end of branch i+1 (in p.u. or V)
    return Ri * (Pi**2 + Qi**2) / (Vi**2)




# --- PARAMETERS ---
NUM_DG = 10
NUM_SC = 6
POP_SIZE = 20

DG_SIZE_MIN = 50
DG_SIZE_MAX = 500
SC_SIZE_MIN = 50
SC_SIZE_MAX = 300

MAX_ITER = 100

# --- LOAD DATA ---
bus_data = pd.read_csv("./data/bus_data.csv")
bus_indices = bus_data['BusNo'].tolist()
PL_list = bus_data['Load_kW'].tolist()  # List of real loads

# --- INITIALIZATION FUNCTION (same as before) ---
def initialize_population(pop_size, num_dg, num_sc, bus_indices, dg_size_min, dg_size_max, sc_size_min, sc_size_max):
    population = []
    for _ in range(pop_size):
        dg_buses = np.random.choice(bus_indices[1:], num_dg, replace=False)
        dg_sizes = np.random.uniform(dg_size_min, dg_size_max, num_dg)
        available_buses = [b for b in bus_indices[1:] if b not in dg_buses]
        sc_buses = np.random.choice(available_buses, num_sc, replace=False)
        sc_sizes = np.random.uniform(sc_size_min, sc_size_max, num_sc)
        individual = []
        for b, s in zip(dg_buses, dg_sizes):
            individual.extend([int(b), float(s)])
        for b, s in zip(sc_buses, sc_sizes):
            individual.extend([int(b), float(s)])
        population.append(individual)
    return np.array(population)

# --- FITNESS & FUZZY FUNCTIONS (your code, see below for improved usage) ---
# ... [functions calc_PDGI, mu_PDGI, calc_PF, mu_PF, calc_APLI, mu_APLI, mu_Vi, mu_V, LPi] ...

def dummy_powerflow(individual):
    """Dummy example for fitness calculation placeholders."""
    # In real code, run a power flow here and extract all required metrics.
    # For now, just random plausible values:
    pdgi = np.random.uniform(0.3, 0.7)
    pf = np.random.uniform(0.85, 1.0)
    apli = np.random.uniform(0.4, 1.1)
    voltages = np.random.uniform(0.92, 1.08, len(PL_list))
    return pdgi, pf, apli, voltages

# --- RUN INITIALIZATION ---
population = initialize_population(POP_SIZE, NUM_DG, NUM_SC, bus_indices, DG_SIZE_MIN, DG_SIZE_MAX, SC_SIZE_MIN, SC_SIZE_MAX)

BEST_FITNESS = -np.inf
BEST_INDIVIDUAL = None

for iter in range(MAX_ITER):
    fitness_list = []
    for individual in population:
        # EXTRACT DG and SC info
        dg_info = individual[:NUM_DG*2]  # [bus, size, bus, size, ...]
        sc_info = individual[NUM_DG*2:]  # [bus, size, bus, size, ...]
        PDG_list = dg_info[1::2]  # Sizes only
        # Use your actual load list from bus_data for PL_list
        
        # Example: Run powerflow (use your own or dummy as here)
        pdgi, pf, apli, voltages = dummy_powerflow(individual)
        
        # Fuzzy memberships
        mu_pdgi = mu_PDGI(pdgi)
        mu_pf = mu_PF(pf)
        mu_apli = mu_APLI(apli)
        mu_v = mu_V(voltages)
        
        J_F = mu_pdgi + mu_pf + mu_apli + mu_v
        fitness_list.append(J_F)
        
        if J_F > BEST_FITNESS:
            BEST_FITNESS = J_F
            BEST_INDIVIDUAL = individual

    # ---- TODO: Implement your actual GOA update here! ----
    # For now, just reinitialize randomly for demonstration:
    # population = initialize_population(POP_SIZE, NUM_DG, NUM_SC, bus_indices, DG_SIZE_MIN, DG_SIZE_MAX, SC_SIZE_MIN, SC_SIZE_MAX)

# --- SAVE BEST SOLUTION TO CSV FILE (Location-wise) ---
NUM_DG = 10
NUM_SC = 6

# Convert numpy array to list if needed
if isinstance(BEST_INDIVIDUAL, np.ndarray):
    best_solution = BEST_INDIVIDUAL.tolist()
else:
    best_solution = BEST_INDIVIDUAL

# Prepare data for CSV with location
dg_data = []
sc_data = []

# Helper: Map bus number to location name
bus_to_loc = dict(zip(bus_data['BusNo'], bus_data['Location']))

for i in range(NUM_DG):
    bus = int(best_solution[2*i])
    size = best_solution[2*i+1]
    location = bus_to_loc.get(bus, "Unknown")
    dg_data.append({'Type': 'DG', 'BusNo': bus, 'Location': location, 'Size_kW_or_kVAR': size})

for i in range(NUM_SC):
    bus = int(best_solution[2*NUM_DG + 2*i])
    size = best_solution[2*NUM_DG + 2*i + 1]
    location = bus_to_loc.get(bus, "Unknown")
    sc_data.append({'Type': 'Capacitor', 'BusNo': bus, 'Location': location, 'Size_kW_or_kVAR': size})

# Combine and save
combined_data = dg_data + sc_data
df = pd.DataFrame(combined_data)
df.to_csv('./data/DG_SC_placement_locationwise.csv', index=False)