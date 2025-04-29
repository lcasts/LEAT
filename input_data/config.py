### This file includes all configs to execute main.py properly

#region: #### General Config ####
#### Imports ####
#-------------------------------------
import numpy as np

#### Choose which Scenarios to process ####
#-------------------------------------
all_scenarios = False       # When False only the user_defined_scenarios will be processed
user_defined_scenarios = ["2023-010_Starlink5277"]       #2023-010_Starlink5277      # Based on the scenario name
use_external_trajectory_data = True
file_name_ext_trajectory = 'input_data/trajectory_results_cleaned.csv'
#endregion

#region: #### Trajectory Config ####
#### Simulation Parameters ####
#-------------------------------------
t_max = 720                 # Maximum simulation time [s]
t_span = (0, t_max)         # Time span of the simulation
numeric_method = 'RK45'     # Numerical calculation method to be used 
                            # Possible numeric solvers: 'RK45', 'RK23', 'DOP853', 'BDF'
max_step = 0.1              # Step size [s]
atol = 1                    # Absolute tolerance
rtol = 1                    # Relative tolerance
                            # Higher tolerances lead to faster computation but potentially less accurate results, while lower tolerances increase accuracy but may require more computation time.
use_ode_break = True        # Choose if you want to include an ODE Break Event based on height
height_ode_break = 100e3    # Maximum Simulation Height [m]

#### Constants and Variables ####
#-------------------------------------
R_0 = 6.378388e6            # Mean Earth Radius [m] (WGS84)
g_0 = 9.807                 # Mean Earth Gravity [m/s²]
R = 287.058                 # J/(kg·K)      for p = ρ⋅R⋅T
kappa = 1.4                  # Ratio of specific heats (Cp/Cv) for air

#### Initial Values (Trajectory ODE) ####
#-------------------------------------
v_0 = 0                     # Initial state: Speed without earth rotation
x_0 = 0                     # Initial state: Horizontal distance [m]
s_0 = 0                     # Initial state: Total distance [m]
gamma_0 = 90*np.pi/180      # Path angle relative to horizon initially at 90° (vertical start)

#### Data Compression ####
#-------------------------------------
compress_data = True                # For faster emission calculation time with NASA CEA
compress_method = "height"            # Choose between "time" or "height" as the compression interval
#compress_method = "height"
compress_interval = 1              # Choose interval steps in [s] or [km]
compress_atmosphere = "averages"    # Choose "averages" to average the atmosphere data of the interval
#compress_atmosphere = "latest"     # Or choose "latest" for just the value at the given time/height step
interval_tolerance = 1e-6           # Allow for floating-point precision issues

#endregion

#region: #### Emissions Config ####
#### General Emission Config ####
#-------------------------------------
# Choose if you want to include emission calculation to trajectory calculation
calculate_emissions = True

# Choose which emission calculation type to use
use_emission_factors = True
emission_factor_method = "stoichiometric"
use_nasa_cea = False
use_cantera = False

# Choose if you want to include Black Carbon in the results
calculate_black_carbon = True

#### Emission Constants ####
#-------------------------------------
# Molar masses of the atm species in kg/mol
molar_masses = {
    'N2': 28.0134e-3,  # kg/mol
    'O2': 31.9988e-3,  # kg/mol
    'O': 16.00e-3,     # kg/mol
    'He': 4.002602e-3, # kg/mol
    'H': 1.00784e-3,   # kg/mol
    'Ar': 39.948e-3,   # kg/mol
    'N': 14.0067e-3,   # kg/mol
    #'aox': 16.00e-3,   # Anomalous oxygen, assume same as O
    'NO': 30.0061e-3   # kg/mol
}

# Avogadro's number
avogadro_number = 6.02214076e23  # molecules per mole

# Threshold for species to be considered not zero
threshold = 1e-6


#### NASA CEA Config ####
#-------------------------------------
trace = "1.e-10"

# Rocket Problem (Both can be True -> runs twice)
rocket_equilibrium = True           # Use equilibrium in rocket problem
rocket_frozen = True                # Use frozen in rocket problem
frozen_nfz = 2                      # Frozen Factor

frozen_nfz_exception_fuels = ["HTPB", "PBAN"]   # Fuels with exception nfz value
frozen_nfz_exception = 3                        # Exception nfz value

# Afterburning
afterburning = True                 # Include afterburning in results
problem_afterburning = "hp"         # tp, hp & uv (choose only one)
rof_f = 4                           # Afterburning diameter factor

# Fuels with specific formula to use instead of data from thermo.lib
# When the fuel is specified here, this data will always be used instead of the thermo.lib data
special_fuels = {
    "HTPB": {
        "enthalpy": -19050.8338,  # Enthalpy in kJ/mol
        "chemical_formula": {
            "C": 656,  # Carbon
            "H": 978,  # Hydrogen
            "O": 13,   # Oxygen
            "N": 5     # Nitrogen
        }
    },
    "OtherFuel": {                # Example Fuel
        "enthalpy": -12345.6789,  # Enthalpy in kJ/mol
        "chemical_formula": {
            "C": 300,  # Carbon
            "H": 400,  # Hydrogen
            "O": 50    # Oxygen
        }
    }
    # Add more fuels as needed
}

#endregion

#region: #### Folder Paths, Data Names & Co. ####
# Folders
input_data_folder_path = 'input_data/'
output_data_folder_path_trajectory = 'output_data/trajectory'
output_data_folder_path_nasa_cea = 'output_data/emissions'
output_data_folder_path_excel = 'output_data/data_raw/excel/'
output_data_folder_path_raw = 'output_data/data_raw/txt/'
nasa_cea_folder_path = r'NASA_CEA\\'

# Tags
output_data_trajectory_name = "TRAJ_"
output_data_emissions_name = "EMIS_"
output_data_raw_name = "RAW_"
output_data_compression_name = "COMP_"

# Filenames
nasa_cea_exe = 'FCEA2.exe'
file_name_emission_factors = 'emission_factors.xlsx'
sheet_name_emission_factors = 'Emission_Factors'

# file_name_launch_rates = 'launch_rates.xlsx'
# sheet_name_launch_rates = 'Startraten'

file_name_scenarios = 'scenarios.xlsx'
sheet_name_scenarios = 'scenarios'

file_name_launch_sites = 'launch_sites.xlsx'
sheet_name_launch_sites = 'launch_sites'

file_name_launch_vehicle = 'LV_Database.xlsx'
sheet_name_launch_vehicle = 'LV_Database'
sheet_name_launch_vehicle_engines = 'Engine'


#### Colors for Terminal Outputs ####
class colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

#endregion