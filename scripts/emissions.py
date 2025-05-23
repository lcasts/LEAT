### This file includes all functions to process the data with nasa cea

#region: Imports & Co.
from input_data.config import *
import subprocess
import shutil
import os
import re
import math
import warnings
import pandas as pd
import cantera as ct
from scripts import data_processing
from scripts import trajectory
from ast import literal_eval
from scipy.optimize import fsolve

# Suppress warnings from openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#endregion

#region: Additional subfunctions to process the two main functions
# Generating an engine dataframe for first NASA CEA calculations
def create_engine_df(df_phases, df_launch_vehicles_engines):
    # Function to convert fuel mass to mass fractions
    def to_mass_fraction(mass_list):
        if isinstance(mass_list, list) and mass_list:
            total_mass = sum(mass_list)
            return [mass / total_mass for mass in mass_list]
        else:
            return 0
    
    # Filtering Phases Dataframe
    df_phases_filtered = df_phases.loc[:, ["booster_engine", "booster_fuel", "booster_fuel_temp", "booster_fuel_mass", "stage_engine", "stage_fuel", "stage_fuel_temp", "stage_fuel_mass"]]
    
    # Reset the index to include the phase as a column
    df_phases_filtered.reset_index(inplace=True)
    df_phases_filtered.rename(columns={"index": "phase"}, inplace=True)
    
    # Change mass to massfractions
    df_phases_filtered['booster_fuel_mass'] = df_phases_filtered['booster_fuel_mass'].apply(to_mass_fraction)
    df_phases_filtered['stage_fuel_mass'] = df_phases_filtered['stage_fuel_mass'].apply(to_mass_fraction)

    # Create the combined DataFrame
    engine_df = []
    for idx, row in df_phases_filtered.iterrows():
        if pd.notna(row["booster_engine"]):
            engine_df.append([row["phase"], row["booster_engine"], row["booster_fuel"], row["booster_fuel_temp"], row["booster_fuel_mass"]])
        if pd.notna(row["stage_engine"]):
            engine_df.append([row["phase"], row["stage_engine"], row["stage_fuel"], row["stage_fuel_temp"], row["stage_fuel_mass"]])

    engine_df = pd.DataFrame(engine_df, columns=["phase", "engine", "fuel", "fuel_temp", "fuel_mf"])
    
    # Merge with engine data from df_launch_vehicles_engines
    engine_df = pd.merge(engine_df, df_launch_vehicles_engines, left_on='engine', right_index=True, how='left')

    return engine_df

# Function to create the NASA CEA .inp file
def create_inp_file(problem, filename, rocket_case=None, df=None, *args):
    def generate_chemical_formula_string(chemical_formula):
        return ' '.join([f"{element} {amount}" for element, amount in chemical_formula.items()])
    
    def create_inp_string_rocket(df, case_string):
        # Check for exception fuels in the dataframe
        has_exception_fuel = any(fuel in frozen_nfz_exception_fuels for fuel in df['fuel'])
        
        # Adjust nfz value if exception fuel is present
        if "frozen" in case_string and has_exception_fuel:
            case_string = f"frozen nfz={frozen_nfz_exception}"
        elif "frozen" in case_string:
            case_string = f"frozen nfz={frozen_nfz}"
        
        
        inp_string = "problem \n"
        inp_string += f"  rocket {case_string} \n"
        #inp_string += "  rocket equilibrium tcest,k={} \n".format(df.loc[i, 'tcest'])
        inp_string += "  p,(bar)={}, \n".format(df['p_chamber'])
        inp_string += "  sup,ae/at={}, \n".format(df['Ae/At'])
        #if nozzle == True:
        #    inp_string += "  sup,ae/at={} \n".format(df.loc[i, 'nozzle_expansion_ratio'])
        #if pressure == True:
        #    inp_string += "  pip={} \n".format(df.loc[i, 'chamber_pressure'][k]/df.loc[i, 'p_end'])      
                      
        inp_string += "react \n"
        for j in range(len(df['fuel'])):
            fuel_name = df['fuel'][j]
            fuel_mass_fraction = df['fuel_mf'][j]
            fuel_temp = df['fuel_temp'][j]
            
            # Check if the current fuel is in special_fuels
            if fuel_name in special_fuels:
                # Retrieve the special fuel's data
                fuel_data = special_fuels[fuel_name]
                enthalpy = fuel_data["enthalpy"]
                chemical_formula = fuel_data["chemical_formula"]
                formula_str = generate_chemical_formula_string(chemical_formula)
                
                # Include detailed fuel definition for special fuels
                inp_string += (
                    f"  name={fuel_name} wt={fuel_mass_fraction} t,k={fuel_temp} "
                    f"h,kj/mol={enthalpy} {formula_str} \n"
                )
            else:
                # Standard definition for non-special fuels
                inp_string += f"  name={fuel_name} wt={fuel_mass_fraction} t,k={fuel_temp} \n"

        inp_string += "output  \n"
        #inp_string += "  siunits massf \n"
        inp_string += "  siunits massf trace={} \n".format(trace)
        #inp_string += "  siunits massf transport \n"
        inp_string += "  plot p t mach H rho s \n"
        inp_string += "end \n"
        
        return inp_string

    def create_inp_string_afterburning(*args):
        line = "problem o/f={},\n".format(rof_ab)
        
        #  Input based on problem type - values for equilibrium at the exit #
        if problem_afterburning == "tp":
            line += "  " + problem_afterburning + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
        elif problem_afterburning == "hp":
            line += "  " + problem_afterburning + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
        elif problem_afterburning == "uv":
            rho_comb = rho_fuel*(1/(1+rof_ab)) + rho_ab*(rof_ab/(1+rof_ab))
            line += "  " + problem_afterburning + "  t,k={} rho,kg/m**3={} \n".format(t_ab, rho_comb)
        # elif problem_afterburning == "sv":
        #     line += "  " + problem_afterburning + "  s/r={} rho,kg/m**3={} \n".format(s_ab/8.31451, rho_ab)
        # elif problem_afterburning == "sp":
        #     line += "  "+ problem_afterburning + "  s/r={} p,bar={} \n".format(s_ab/8.31451, p_ab)
        
        # Separate fuel and air species into dictionaries with their mass fractions
        fuel_species = {species: df_fuel_massfractions[species].values[0] for species in df_fuel_massfractions.columns}
        air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

        # Check and limit total species count
        total_species = len(fuel_species) + len(air_species)

        if total_species > 24:
            # Drop *He and *Ar from air species if total species are between 24 and 26
            air_species = {k: v for k, v in air_species.items() if k not in ["He", "Ar"]}
            total_species = len(fuel_species) + len(air_species)

            # If still exceeding 24, drop the smallest fuel species by mass fraction
            if total_species > 24:
                fuel_species = dict(sorted(fuel_species.items(), key=lambda x: x[1], reverse=True)[:24 - len(air_species)])

        # Include Fuel and Air Species
        line += "react \n"
        for fuel, mass_fraction in fuel_species.items():
            line += "  fuel={} wt={} t,k={} \n".format(fuel, mass_fraction, t_fuel)

        for air, mass_fraction in air_species.items():
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, t_air)
                
        # Ending Lines
        line += "output \n"
        line += "  siunits massf trace={} \n".format(trace)
        line += "  plot p t mach H rho s \n"
        line += "end \n"
        
        return line

    if problem == "rocket":
        if rocket_case == "equilibrium":
            case_string = "equilibrium"
        elif rocket_case == "frozen":
            case_string = "frozen"
            
        inp_string = create_inp_string_rocket(df, case_string)

        with open(filename + ".inp", "w") as file:
            file.write(inp_string)
            
    elif problem == "afterburning":
        t_ab, p_ab, rho_ab, s_ab, t_fuel, rho_fuel, t_air, rof_ab, df_fuel_massfractions, df_air_massfractions = args
        
        inp_string = create_inp_string_afterburning()
        
        with open(filename + ".inp", "w") as file:
            file.write(inp_string)

def save_output_files_cantera(df_output, filename, s_name):
    # Convert dictionary to a string for .txt file
    output_txt = "Temperature: {}\nPressure: {}\nEmissions:\n".format(
        df_output['Temperature'], df_output['Pressure'])
    for species, value in df_output['Emissions'].items():
        output_txt += "{}: {}\n".format(species, value[0])

    # Save .txt file
    base_name_txt = filename + "_" + s_name
    new_filename_txt, file_path_txt = data_processing.get_unique_filename(base_name_txt, output_data_folder_path_raw, ".txt")
    with open(file_path_txt, "w") as txt_file:
        txt_file.write(output_txt)
    shutil.move(file_path_txt, os.path.join(output_data_folder_path_raw, new_filename_txt))

    # Convert dictionary to a pandas DataFrame for .xlsx file
    emissions_data = {
        'Species': list(df_output['Emissions'].keys()),
        'Emissions Value': [value[0] for value in df_output['Emissions'].values()]
    }
    df_emissions = pd.DataFrame(emissions_data)

    # Adding temperature and pressure as separate columns
    df_emissions['Temperature'] = df_output['Temperature']
    df_emissions['Pressure'] = df_output['Pressure']

    # Save .xlsx file
    base_name_excel = filename + "_" + s_name
    new_filename_excel, file_path_excel = data_processing.get_unique_filename(base_name_excel, output_data_folder_path_excel, ".xlsx")
    with pd.ExcelWriter(file_path_excel, engine='openpyxl') as writer:
        df_emissions.to_excel(writer, sheet_name=new_filename_excel)

    shutil.move(file_path_excel, os.path.join(output_data_folder_path_excel, new_filename_excel))

# Copy necessary files to run NASA CEA in main directory
def init_directory():
    # Preparation
    # Copy thermo.lib & trans.lib, because they are necessary in main directory for NASA CEA
    files_to_copy = ["thermo.lib", "trans.lib"]
    for file in files_to_copy:
        try:
            shutil.copy("NASA_CEA" + "/" + file, ".")
        except FileNotFoundError:
            pass

def calculate_reactant_enthalpy(fuel, oxidizer, equivalence_ratio, fuel_temp, oxidizer_temp, pressure):
    # Konvertiere Temperaturen und Druck zu float, falls nötig
    fuel_temp = float(fuel_temp)
    oxidizer_temp = float(oxidizer_temp)
    pressure = float(pressure)

    # Ausgabe der Werte und ihrer Datentypen
    
    gas = ct.Solution('gri30.yaml')
    
    # Set the mixture equivalence ratio
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)

    # Get the fuel and oxidizer mass fractions
    fuel_fraction = gas.mole_fraction_dict()[fuel]
    oxidizer_fraction = gas.mole_fraction_dict()[oxidizer]

    # Create separate states for fuel and oxidizer
    fuel_gas = ct.Solution('gri30.yaml')
    fuel_gas.TPX = fuel_temp, pressure, {fuel: 1.0}

    oxidizer_gas = ct.Solution('gri30.yaml')
    oxidizer_gas.TPX = oxidizer_temp, pressure, {oxidizer: 1.0}

    # Calculate the mixture enthalpy as a weighted average
    total_mass = fuel_fraction + oxidizer_fraction
    
    mixture_enthalpy = (
        fuel_fraction / total_mass * fuel_gas.h +
        oxidizer_fraction / total_mass * oxidizer_gas.h
    )

    return mixture_enthalpy

def mach_number(expansion_ratio):
    def equation(Me):
        return (( (kappa+1)/2 ) ** (-(kappa+1)/(2*(kappa-1)))) * (1 + Me**2 * (kappa-1)/2) ** ((kappa+1)/(2*(kappa-1))) - expansion_ratio
    
    # Initial guess for Me (subsonic and supersonic solutions)
    Me_guess = [2.0]  # One subsonic and one supersonic guess
    try:
        solutions = [fsolve(equation, guess)[0] for guess in Me_guess]
    except Exception as e:
        print(f"Error in fsolve for supersonic solution: {e}")
        return None
    
    # Filter only valid positive solutions
    solutions = [sol for sol in solutions if sol > 0]
    return solutions

def combustion_chamber(fuel, oxidizer, rof, pressure, enthalpy):
    # Check the fuel type and select the appropriate mechanism
    MW_H2 = 2.016    # Hydrogen (H2)
    MW_CH4 = 16.04   # Methane (CH4)
    MW_O2 = 32.00    # Oxygen (O2)
    gri30_species = set(ct.Solution('gri30.yaml').species_names)

    if fuel == "H2":
        gas = ct.Solution('h2o2.yaml')
        mass_ratio_H2_O2_real = rof
        mass_ratio_H2_O2_stoich = 7.9366827390519274
        molar_ratio_H2_O2_real = mass_ratio_H2_O2_real * (MW_H2 / MW_O2)
        molar_ratio_H2_O2_stoich = mass_ratio_H2_O2_stoich * (MW_H2 / MW_O2)
        equivalence_ratio = molar_ratio_H2_O2_stoich / molar_ratio_H2_O2_real

    elif fuel == "CH4":
        gas = ct.Solution('gri30.yaml')
        mass_ratio_CH4_O2_real = rof
        mass_ratio_CH4_O2_stoich = 3.989263492008084
        molar_ratio_CH4_O2_real = mass_ratio_CH4_O2_real * (MW_CH4 / MW_O2)
        molar_ratio_CH4_O2_stoich = mass_ratio_CH4_O2_stoich * (MW_CH4 / MW_O2)
        equivalence_ratio = molar_ratio_CH4_O2_stoich / molar_ratio_CH4_O2_real
    else:
        raise ValueError("Unsupported fuel type. Supported fuels: H2, CH4, RP-1 (C12H24).")
  
    # Set initial temperature
    gas.TP = 3000, pressure  # Set temperature to 3000K and pressure
    # Set equivalence ratio and conditions
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    gas.HP = enthalpy, pressure  # Set enthalpy and pressure to define conditions
    gas.equilibrate('HP')
    
    # Extract emissions
    emissions = {species: gas.Y[gas.species_index(species)] for species in gas.species_names}

    return gas

def nozzle_expansion(gas, throat_area, exit_area):
    # Initial conditions from the combustion chamber
    T_0 = gas.T
    P_0 = gas.P
    rho_0 = P_0 / (R * T_0)
    
    # Calculate Gamma using the provided equation
    Gamma = (kappa ** 0.5) * ((2 / (kappa + 1)) ** ((kappa + 1) / (2 * (kappa - 1))))
    
    # Throat conditions (Mach = 1)
    T_throat = T_0 * (1 + ((kappa - 1) / 2) * 1**2) ** -1
    P_throat = P_0 * (1 + ((kappa - 1) / 2) * 1**2) ** (-kappa / (kappa - 1))
    rho_throat = rho_0 * (1 + ((kappa - 1) / 2) * 1**2) ** (-1 / (kappa - 1))
    v_throat = (kappa * R * T_throat) ** 0.5
    mass_flow = rho_throat * v_throat * throat_area
    
    # Calculate exit pressure using area ratio equation
    area_ratio = exit_area / throat_area
    
    # Calculate exit Mach number using isentropic relation
    M_exit = mach_number(area_ratio)[0]
    
    # Calculate exit temperature and density
    T_exit = T_0 * (1 + ((kappa - 1) / 2) * M_exit**2) ** -1
    P_exit = P_0 * (1 + ((kappa - 1) / 2) * M_exit**2) ** (-kappa / (kappa - 1))
    rho_exit = rho_0 * (1 + ((kappa - 1) / 2) * M_exit**2) ** (-1 / (kappa - 1))
    v_exit = M_exit * (kappa * R * T_exit) ** 0.5
    
    # Set gas properties and equilibrate
    throat_gas = gas
    throat_gas.TP = T_throat, P_throat
    throat_gas.equilibrate('SP')
    exit_gas = throat_gas
    exit_gas.TP = T_exit, P_exit
    exit_gas.equilibrate('SP')
    throat_props = {
        'Mach': 1,
        'Temperature': T_throat,
        'Pressure': P_throat,
        'Density': rho_throat,
        'Velocity': v_throat
    }
    
    exit_props = {
        'Mach': M_exit,
        'Temperature': T_exit,
        'Pressure': P_exit,
        'Density': rho_exit,
        'Velocity': v_exit
    }
    
    # Emissions calculation
    emissions = {species: gas.Y[gas.species_index(species)] for species in gas.species_names}
    return {
        'Throat': throat_props,
        'Exit': exit_props,
        'ExitGas': exit_gas,
        'Emissions': emissions
    }

def atmosphere_mixing(t_ab, p_ab, rho_ab, s_ab, t_fuel, t_air, exit_gas, df_air_massfractions, rof_ab):
    # **Step 1: Extract Exit Conditions**
    exit_temperature = t_ab
    exit_pressure = p_ab

    # **Step 2: Map Species Names to Cantera Convention**
    species_mapping = {
        '*N2': 'N2', '*O2': 'O2', '*O': 'O', '*He': 'He',
        '*H': 'H', '*Ar': 'Ar', '*N': 'N', '*NO': 'NO'
    }
    
    # Convert DataFrame species names using mapping
    ambient_mass_fractions = {
        species_mapping.get(species, species): value
        for species, value in df_air_massfractions.iloc[0].items()
        if species_mapping.get(species, species) in ct.Solution('gri30.yaml').species_names
    }
    new_total_mass_fraction = sum(ambient_mass_fractions.values())

    # Normalize mass fractions so they sum to 1
    if new_total_mass_fraction > 0:
        normalized_air_mass_fractions = {
            species: value / new_total_mass_fraction
            for species, value in ambient_mass_fractions.items()
        }
    else:
        normalized_air_mass_fractions = ambient_mass_fractions  # Avoid division by zero

    
    # **Step 3: Calculate Ambient Mass Flow Rate Using rof_ab**
    massflow_exit = sum(exit_gas.values())
    massflow_air = massflow_exit * rof_ab  # Calculate the air mass flow rate
    total_massflow = massflow_exit + massflow_air

    # Scale ambient mass fractions by massflow_air
    scaled_ambient_mass_fractions = {species: value * massflow_air for species, value in normalized_air_mass_fractions.items()}

    # **Step 4: Normalize Mass Fractions**
    mixed_mass_fractions = {
        species: (exit_gas.get(species, 0) * massflow_exit + scaled_ambient_mass_fractions.get(species, 0)) / total_massflow
        for species in set(exit_gas.keys()).union(set(normalized_air_mass_fractions.keys()))
    }

    # **Step 5: Set Up Cantera Gas Object**
    try:
        gas = ct.Solution('gri30.yaml')  # Use gri30.yaml instead of air.yaml
        gas.TP = max(max(exit_temperature, t_ab),300), max(exit_pressure, p_ab)
        gas.Y = {species: value for species, value in mixed_mass_fractions.items() if species in gas.species_names}
    except ct.CanteraError as e:
        print(f"Error initializing gas mixture: {e}")
        return None

    # **Step 6: Perform HP Combustion**
    try:
        gas.equilibrate('HP')  # Equilibrate at constant enthalpy and pressure
    except ct.CanteraError as e:
        print(f"Error in HP Combustion: {e}")
        return None

    # **Step 7: Extract Post-Combustion Properties**
    emissions = {species: gas[species].Y for species in gas.species_names if gas[species].Y > 1e-6}

    return {
        'Temperature': gas.T,
        'Pressure': gas.P,
        'Emissions': emissions
    }

# Save files to output data as backup
def save_files(filename, s_name, rocket_case=None):
    filetype_to_move = [".inp", ".out", ".csv", ".plt"]
    for filetype in filetype_to_move:
        try:
            folder = output_data_folder_path_raw
            new_filename = f"{filename}_{s_name}"
            if rocket_case:
                new_filename = f"{filename}_{s_name}_{rocket_case}"
                
            # Adding (counter) and filetype to new_filename
            new_filename, file_path = data_processing.get_unique_filename(new_filename, folder, filetype)
            # Renaming raw data file and moving it to storage folder
            os.rename(filename + filetype, new_filename)
            shutil.move(new_filename, os.path.join(folder, new_filename))
        except FileNotFoundError:
            pass

# Save formated Results to Excel
def save_results_excel(filename, s_name, df_output, rocket_case=None):
    folder = output_data_folder_path_excel
    base_name = filename + "_" + s_name
    if rocket_case:
        base_name = filename + "_" + s_name + "_" + rocket_case
    
    filename, file_path = data_processing.get_unique_filename(base_name, folder, ".xlsx")
    
    with pd.ExcelWriter(folder + filename, engine='openpyxl') as writer:
        df_output.to_excel(writer, sheet_name=filename)
                
# Move and delete all files created by NASA CEA
def clean_directory(filename):
    ## Clean up - Delete (.xlsx, .inp, .out, .lib) in the main directory   
    #filetype_to_remove = [".xlsx", ".inp", ".out", ".csv", ".plt"]
    filetype_to_remove = [".inp", ".out", ".csv", ".plt"]
    for filetype in filetype_to_remove:
        try:
            os.remove(filename + filetype)
        except FileNotFoundError:
            pass
   
# Delete NASA CEA init files
def clean_up():
    # Remove thermo.lib & trans.lib
    files_to_remove = ["thermo.lib", "trans.lib"]
    for file in files_to_remove:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass  

# Function to calculate the rof for afterbuning
def rof_ab_calc(rof_f, dia, mf_engine, rho_atm, v_rocket):
    rof_ab = v_rocket*rho_atm*np.pi*(((dia*rof_f)**2-dia**2)/4)/mf_engine
    return rof_ab

def clean_species_name(species_name):
    return species_name.strip('*').strip()

# Function to read in the plt file
def read_plt_file(plt_filename):
    with open(plt_filename, "r") as file:
        lines = file.readlines()

    header_line = None
    for line in lines:
        if line.startswith("#"):
            header_line = line.strip().lstrip("#").split()
            break

    if header_line is None:
        raise ValueError("No header line found in the .plt file.")

    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

    num_lines = len(data_lines)

    if num_lines == 3:
        row_names = ["chamber", "throat", "exit"]
    elif num_lines == 1:
        row_names = ["exit"]
    else:
        raise ValueError(f"Unexpected number of lines: {num_lines}. Expected: 1 or 3.")

    data = [list(map(float, line.split())) for line in data_lines]

    df = pd.DataFrame(data, columns=header_line)
    df.index = row_names
    df = df.transpose()
    
    return df

# Function to extract data from the .out and .plt file
def readCEA(problem, filename, rocket_case=None):
    # Function to read .out file of rocket problem (eq. and frozen)
    def read_rocket(rocket_case):
        equilibrium_data = []
        frozen_data = []
        start_processing = False
        current_mode = None

        with open(filename + ".out", 'r') as file:
            # Search for Equilibrium Data
            for line in file:
                line = line.strip()

                # Check for the start of the equilibrium section
                if "THEORETICAL ROCKET PERFORMANCE ASSUMING EQUILIBRIUM" in line:
                    current_mode = "equilibrium"
                    start_processing = False  # Ensure processing starts fresh
                    continue
                
                # Check for the start of the frozen section after processing equilibrium section
                if "THEORETICAL ROCKET PERFORMANCE ASSUMING FROZEN COMPOSITION" in line:
                    current_mode = "frozen"
                    start_processing = False  # Ensure processing starts fresh
                    continue
                
                # Start processing when encountering "MASS FRACTIONS"
                if "MASS FRACTIONS" in line:
                    start_processing = True
                    continue

                # Stop processing when encountering "* THERMODYNAMIC PROPERTIES FITTED TO 20000.K"
                if "* THERMODYNAMIC PROPERTIES FITTED TO 20000.K" in line:
                    start_processing = False
                    # Skip further processing in this section and look for the next relevant section
                    break

                if start_processing:
                    if current_mode == "equilibrium":
                        # Use the provided regex to process equilibrium data
                        match = re.match(r"(\S[\S\s]*?)\s+(.{8})\s+(.{8})\s+(.{8})", line)
                        if match:
                            species = match.group(1).strip()
                            chamber_str = match.group(2).strip().replace(" ", "")
                            throat_str = match.group(3).strip().replace(" ", "")
                            exit_str = match.group(4).strip().replace(" ", "")
                            
                            chamber_num = float(re.sub(r"([-+])", r"E\1", chamber_str, count=1))
                            throat_num = float(re.sub(r"([-+])", r"E\1", throat_str, count=1))
                            exit_num = float(re.sub(r"([-+])", r"E\1", exit_str, count=1))
                            
                            equilibrium_data.append([species, chamber_num, throat_num, exit_num])
                            
                    elif current_mode == "frozen":
                        match = re.match(r"(\S[\S\s]*?)\s+([0-9]*\.[0-9]{5})(?:\s+(\S[\S\s]*?)\s+([0-9]*\.[0-9]{5}))?(?:\s+(\S[\S\s]*?)\s+([0-9]*\.[0-9]{5}))?", line)
                        if match:
                            species1 = match.group(1).strip()
                            value1 = float(match.group(2).strip())
                            frozen_data.append([species1, value1])

                            if match.group(3):
                                species2 = match.group(3).strip()
                                value2 = float(match.group(4).strip())
                                frozen_data.append([species2, value2])

                            if match.group(5):
                                species3 = match.group(5).strip()
                                value3 = float(match.group(6).strip())
                                frozen_data.append([species3, value3])
                            

        # Convert data to DataFrames
        df_equilibrium = pd.DataFrame(equilibrium_data, columns=["Species", "chamber", "throat", "exit"]).set_index("Species")
        df_frozen = pd.DataFrame(frozen_data, columns=["Species", "exit"]).set_index("Species")

        if rocket_case == "equilibrium":
            species_df = df_equilibrium
        elif rocket_case == "frozen":
            species_df = df_frozen
        
        # Extract thermodynamic data from .plt
        other_values_df = read_plt_file(filename + ".plt")

        # Combine both DataFrames
        combined_df = pd.concat([other_values_df, species_df])
        return combined_df

    # Function to read .out file of problem types tp, uv, etc.     
    def read_problem():
        # Read .out file
        with open(filename + ".out") as f:
            data = f.read() 
        
        match = re.search(r"MASS FRACTIONS(.*)THERMODYNAMIC PROPERTIES", data, re.DOTALL)
        lines = match.group(1).split("\n")
        lines2 = [line for line in lines if len(line) > 10]
        cleaned_lines = [line.strip().split() for line in lines2]
        mass_fractions_df = pd.DataFrame(columns=['exit'])
        
        for item in cleaned_lines:
            if '*****' in item[1]:
                item[1] = item[1].replace('*****', '1.00')
                print(f"CEA error workaround: Replaced '*****' with '1.00' in item: {item}")
            parts = item[1].split('-') if '-' in item[1] else item[1].split('+')
            result = float(parts[0]) * 10**(-int(parts[1])) if '-' in item[1] else float(parts[0]) * 10**(int(parts[1]))
            new_df = pd.DataFrame({'exit': result}, index=[item[0]])
            mass_fractions_df = pd.concat([mass_fractions_df, new_df])
        mass_fractions_df.index = [item[0] for item in cleaned_lines]
        
        ##### Extract remaining data from .csv #####
        other_values_df = read_plt_file(filename + ".plt")
        
        ##### Combine both df #####
        combined_df = pd.concat([other_values_df, mass_fractions_df])

        return combined_df
    
    # Execute respective function based on problem type
    if problem == "rocket":
        output_df = read_rocket(rocket_case)  
    elif problem != "rocket":
        output_df = read_problem()

    return output_df

# Function to create the current relevant Fuel Dataset (phase, engine, interval_fuel_mass)
def create_fuel_tuple(current_phase, phase, phase_time_interval, time_interval):
    fuel_data = []

    # Check if booster data is available and calculate fuel-related parameters
    if not pd.isna(current_phase["booster_engine"]):
        booster_engine = current_phase["booster_engine"]
        booster_fuel_masses = current_phase["booster_fuel_mass"]
        booster_massflow = sum(booster_fuel_masses) / phase_time_interval
        interval_booster_fuel_mass  = booster_massflow * time_interval
        # Add data to fuel_data list
        fuel_data.append((phase, booster_engine, interval_booster_fuel_mass))

    # Check if stage data is available and calculate fuel-related parameters
    if not pd.isna(current_phase["stage_engine"]):
        stage_engine = current_phase["stage_engine"]
        stage_fuel_masses = current_phase["stage_fuel_mass"]
        stage_massflow = sum(stage_fuel_masses) / phase_time_interval
        interval_stage_fuel_mass = stage_massflow * time_interval
        # Add data to fuel_data list
        fuel_data.append((phase, stage_engine, interval_stage_fuel_mass))
        
    return fuel_data

# Function to Calculate Final Black Carbon Emission Index
def black_carbon(height, factor):
    result = factor * max(0.04, min(1, 0.04 * math.exp(0.12 * (height - 15))))
    
    return result

# Handle NASA CEA Errors
def handle_nasa_cea_error_message():
    pass
    # *******************************************************************************

    #         NASA-GLENN CHEMICAL EQUILIBRIUM PROGRAM CEA2, MAY 21, 2004
    #                 BY  BONNIE MCBRIDE AND SANFORD GORDON
    #     REFS: NASA RP-1311, PART I, 1994 AND NASA RP-1311, PART II, 1996

    # *******************************************************************************



    # problem
    # rocket equilibrium tcest,k=3000
    # p(bar)=115
    # sup,ae/at=58.2
    # reactant
    # name=LH2 wt=0.19323685424354242 t,k=293
    # name=LOX wt=0.8067631457564576 t,k=293
    # output
    # siunits massf transport
    # plot p t mach H rho s
    # end

    # OPTIONS: TP=F  HP=F  SP=F  TV=F  UV=F  SV=F  DETN=F  SHOCK=F  REFL=F  INCD=F
    # RKT=T  FROZ=F  EQL=T  IONS=F  SIUNIT=T  DEBUGF=F  SHKDBG=F  DETDBG=F  TRNSPT=T

    # TRACE= 0.00E+00  S/R= 0.000000E+00  H/R= 0.000000E+00  U/R= 0.000000E+00

    # Pc,BAR =   115.000000

    # Pc/P =

    # SUBSONIC AREA RATIOS =

    # SUPERSONIC AREA RATIOS =    58.2000

    # NFZ=  1  Mdot/Ac= 0.000000E+00  Ac/At= 0.000000E+00

    # YOUR ASSIGNED TEMPERATURE  293.00K FOR LH2            
    # IS OUTSIDE ITS TEMPERATURE RANGE20000.00 TO     0.00K (REACT)

    # ERROR IN REACTANTS DATASET (INPUT)

    # FATAL ERROR IN DATASET (INPUT)

#endregion

#region: Main Functions
#### Emission Factors - Main Function
def emission_factors(df_trajectory_results_cleaned, df_phases, df_emission_factors, scenario_name):
    try:
        # 1. Inits
        # Ensure 'altitude_interval' is parsed as a list of floats
        df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
            lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
        )
        # Copy cleaned ODE Results
        emission_results = df_trajectory_results_cleaned
        # Initialize a DataFrame to hold all species emissions for all rows
        total_emissions = pd.DataFrame()
        
        # (a) Iterate through every "row" of the trajectory dataset
        for idx, row in df_trajectory_results_cleaned.iterrows():
            
            # Filter data
            active_phases = row['active_phases']
            time_intervals = row['time_interval']
            altitude = row['h']
            # Initialize a DataFrame to hold all species emissions for the current row of iteration (a)
            total_row_emissions = pd.DataFrame(0, index=[idx], columns=emission_results.columns)
            

            # (b) Iterate through every "active phase and time interval pair" in the current row
                # Multiple phases can be active within one row of a compressed time interval
            for phase, interval in zip(active_phases, time_intervals):
                
                # Init/Filter data
                current_phase = df_phases.loc[phase]
                interval_start, interval_end = float(interval[0]), float(interval[1])
                # Time interval of the given data set (timestep or compressed time interval)
                time_interval = interval_end - interval_start
                # Time interval of the current phase in iteration (b)
                phase_time_interval = current_phase["time_end"] - current_phase["time_start"]
                # Function to create the current relevant Fuel Dataset (phase, engine, interval_fuel_mass)
                    # A list of tuples to store booster and stage data
                fuel_data = create_fuel_tuple(current_phase, phase, phase_time_interval, time_interval)
                
                # (c) Iterate through every "booster or stage" (in the new dataset fuel_data) for each "active phase and time interval pair" in the current row
                    # Multiple engines (booster or stage) can be active within the phases of iteration (b)
                for phase, engine, interval_fuel_mass in fuel_data:
                    # This part is different from NASA CEA Calculation:
                        # Instead of using the massfractions specific to an engine based on NASA CEA Calculations, just the massfractions from the given Excel Sheet are used for given Fuel Combination
                        
                    # Determine if the engine is a booster or stage engine
                    if engine == current_phase['booster_engine']:
                        current_fuel_type = current_phase['booster_fuel']
                    elif engine == current_phase['stage_engine']:
                        current_fuel_type = current_phase['stage_fuel']
                    else:
                        current_fuel_type = None
                    
                    # Transform the fuel type list into a single string with slashes
                    current_fuel_type_str = "/".join(current_fuel_type)

                    # Filter `df_emission_factors` by fuel type and method
                    df_emission_factor_fuel = df_emission_factors[
                        (df_emission_factors['fuel'] == current_fuel_type_str) &
                        (df_emission_factors['method'] == emission_factor_method)
                    ]
                    
                    # Further filter by altitude interval
                    altitude_interval_match = df_emission_factor_fuel[
                        df_emission_factor_fuel['altitude_interval'].apply(
                            lambda x: x[0] <= altitude <= x[1] if isinstance(x, list) else False
                        )
                    ]

                    # Select the first matching row of emission factors
                    emission_factors_row = altitude_interval_match.iloc[0]

                    # Create a copy of `emission_factors_row` to work with for calculation
                    df_emission_factor_fuel = emission_factors_row.copy()

                    # Skip if no matching altitude interval
                    if altitude_interval_match.empty:
                        print(f"No matching emission factors for altitude {altitude}")
                        continue

                    # Update Black Carbon Primary to Final Emission Index
                    if calculate_black_carbon:
                        primary_ei_bc = df_emission_factor_fuel['BC_prim']
                        final_ei_bc = black_carbon(row["h"], primary_ei_bc)
                        
                        # Update the BC value in the DataFrame with the new value
                        df_emission_factor_fuel['BC_sec'] = final_ei_bc
                        
                    # Filter to keep only numeric columns for multiplication
                    df_emission_factor_fuel_numeric = df_emission_factor_fuel.drop(['fuel', 'method', 'altitude_interval','BC_prim']).astype(float)

                    # Multiply mass fractions with interval_fuel_mass to calculate emissions for each species
                    df_species_mass = df_emission_factor_fuel_numeric * interval_fuel_mass
                    df_species_mass = df_species_mass.fillna(0)
                    
                                                      
                    # (d) Iterate through all species of current engine in iteration (c) and add them to the total_row_emissions DataFrame
                    for species, mass in df_species_mass.items():
                        # Check if species column exists in total_row_emissions and accumulate values
                        if species in total_row_emissions.columns:
                            total_row_emissions.loc[idx, species] += mass
                        else:
                            total_row_emissions.loc[idx, species] = mass

            # Append total_row_emissions to total_emissions after iteration (b), (c) and (d) are finished
            total_emissions = pd.concat([total_emissions, total_row_emissions])
            total_emissions = total_emissions.fillna(0)
        
        # Combine total_emissions into emission_results, while dropping the initial non-species columns of total_emissions
        emission_results = pd.concat([emission_results, total_emissions.iloc[:, 7:]], axis=1)
        
    except Exception as e:
        data_processing.handle_error(e, "emission_factors", "Error in Emission Factors main function.")
        
    return emission_results

#### Cantera - Main Function
def run_cantera(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors):
    try:
        # 1. Inits 
        # Copy thermo.lib, trans.lib, cleaned ODE Results and create engine df
        init_directory()
        emission_results = df_trajectory_results_cleaned
        df_engines = create_engine_df(df_phases, df_launch_vehicles_engines)
        print(colors.BOLD + colors.BLUE + "Calculating Emissions using Cantera..." + colors.END)

        # 2. Rocket Problem - Engines
        # 2.1 Run Cantera Rocket solver for every Engine (Booster or Stage)
        for idx, row in df_engines.iterrows():
            # Unique filenames for directory management
            filename = scenario_name + "_" + row["engine"]
            s_name = row["phase"] + "_rocket"
            fuel = row['fuel'][0].replace('(L)', '').strip()
            oxidizer = row['fuel'][1].replace('(L)', '').strip()
            fuel_mf = row['fuel_mf'][0]
            oxidizer_mf = row['fuel_mf'][1]
            fuel_temp = row['fuel_temp'][0]
            oxidizer_temp = row['fuel_temp'][1]
            equivalence_ratio =  oxidizer_mf/fuel_mf
            chamber_pressure = row['p_chamber'] * 1e5
            fuel_temp = oxidizer_temp = reaction_temp = 3000 # Impact on reaction
            reactant_enthalpy = calculate_reactant_enthalpy(fuel, oxidizer, equivalence_ratio, fuel_temp, oxidizer_temp, chamber_pressure)

            # Step 1: Combustion chamber
            combustion_gas = combustion_chamber(fuel, oxidizer, equivalence_ratio, chamber_pressure, reactant_enthalpy)
            d_exit = row['d_exit']
            exit_area = (row['d_exit'])**2*(1/4)*np.pi
            throat_area = exit_area/(row['Ae/At'])

            # Step 2: Nozzle expansion
            df_output = nozzle_expansion(combustion_gas, throat_area, exit_area)
            exit_temperature = df_output['Exit']['Temperature']
            exit_pressure = df_output['Exit']['Pressure'] 

            # Fix key capitalization for emissions
            emissions = df_output.get('Emissions', {})

            # Add exit pressure and exit temperature to df_engines
            df_engines.loc[idx, 'exit_pressure'] = exit_pressure
            df_engines.loc[idx, 'exit_temperature'] = exit_temperature
            
            # Filter df_output and add species exit massfractions to df_engines
            df_species_filtered = emissions
            # Loop through each species in the filtered DataFrame
            for species, value in df_species_filtered.items():
                if value > threshold:
                    # Update the DataFrame with the value if it exceeds the threshold
                    df_engines.loc[idx, species] = value
                else:
                    # Set species value to None if below threshold
                    if species in df_engines.columns:
                        df_engines.loc[idx, species] = None

            # Update general nozzle exit conditions
            exit_conditions = df_species_filtered.get('Exit', {})
            for key, value in exit_conditions.items():
                if key in df_engines.columns:
                    df_engines.loc[idx, key] = value

            # Saving raw and formated data
            save_files(filename, s_name)
            clean_directory(filename)
            #save_results_excel(filename, s_name, df_output)
        
        # 2.2 Save Results at Exit for all Engines to Excel and copy original data for subsequent Afterburning Calculations
        df_engines = df_engines.fillna(0)
        save_results_excel(scenario_name, "Rocket_Exit_Results", df_engines)
        df_engines_original_data = df_engines.copy()
        
        # 3. Total Emission Calculation
        # Initialize a DataFrame to hold all species emissions for all rows
        total_emissions = pd.DataFrame()

        # (a) Iterate through every "row" of the trajectory dataset
        for idx, row in emission_results.iterrows():
            # Filter data
            active_phases = row['active_phases']
            time_intervals = row['time_interval']
            altitude = row['h']
            # Initialize a DataFrame to hold all species emissions for the current row of iteration (a)
            total_row_emissions = pd.DataFrame(0, index=[idx], columns=emission_results.columns)
            # (b) Iterate through every "active phase and time interval pair" in the current row
                # Multiple phases can be active within one row of a compressed time interval
            for phase, interval in zip(active_phases, time_intervals):
                
                # Init/Filter data
                current_phase = df_phases.loc[phase]
                interval_start, interval_end = float(interval[0]), float(interval[1])
                # Time interval of the given data set (timestep or compressed time interval)
                time_interval = interval_end - interval_start
                # Time interval of the current phase in iteration (b)
                phase_time_interval = current_phase["time_end"] - current_phase["time_start"]
                # Function to create the current relevant Fuel Dataset (phase, engine, interval_fuel_mass)
                    # A list of tuples to store booster and stage data
                fuel_data = create_fuel_tuple(current_phase, phase, phase_time_interval, time_interval)
                
                # (c) Iterate through every "booster or stage" (in the new dataset fuel_data) for each "active phase and time interval pair" in the current row
                    # Multiple engines (booster or stage) can be active within the phases of iteration (b)
                for phase, engine, interval_fuel_mass in fuel_data:

                    # Including Afterburning calculation if required
                    if afterburning:
                        # 01. Get required Data for NASA CEA Calculation
                        # 01.1 Athmospheric Data:
                            # at current position (from row, either average or last specified in config)
                        rho_atm, t_atm, p_atm, df_air_massfractions = data_processing.calculate_atmosphere_data(row)
                        dia = df_launch_vehicles['d'].iloc[0]
                        massflow, mdot, dmdot = trajectory.calculate_massflow(current_phase)
                        v_rocket = row["v"]
                        rof_ab = rof_ab_calc(rof_f, dia, massflow, rho_atm, v_rocket)

                        # 01.2 Exhaust data:
                            # at exit from first engine rocket problem NASA CEA calculation (2.1)
                            # Access the original exit data of current "phase, engine pair" from df_engines_original_data
                        df_fuel_massfractions = df_engines_original_data.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines_original_data.columns[df_engines_original_data.columns.get_loc('exit_temperature')+1:]]
                                               
                        # Remove the unnecessary species from the dataset
                        for species in df_fuel_massfractions.columns:
                            # Get the value of the species
                            value = df_fuel_massfractions[species].values[0]
                            
                            # Check if the value is below or equal to the threshold
                            if value <= threshold:
                                # Drop the species column if the value is below or equal to the threshold
                                df_fuel_massfractions.drop(columns=[species], inplace=True)

                        fuel_exit_temperature = df_engines_original_data.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), 'exit_temperature'].values[0]
                        
                        # 01.3 Problem Type Data:
                        t_ab = t_atm
                        p_ab = p_atm*1e-5
                        rho_ab = rho_atm
                        s_ab = None
                        t_fuel = fuel_exit_temperature
                        t_air = t_atm

                        # 02. Cantera Afterburning Calculation of Exhaust and Atmosphere for current "phase, engine pair"
                        afterburning_filename = scenario_name + "_" + str(interval_start)
                        s_name = phase + "_" + engine + "_afterburning"
                        exit_gas = emissions
                        ambient_gas = df_air_massfractions
                        df_output = atmosphere_mixing(t_ab, p_ab, rho_ab, s_ab, t_fuel, t_air, exit_gas, ambient_gas, rof_ab)
                        species_mapping = {
                            '*N2': 'N2', '*O2': 'O2', '*O': 'O', '*He': 'He',
                            '*H': 'H', '*Ar': 'Ar', '*N': 'N', '*NO': 'NO'
                        }
                        
                        # Convert DataFrame species names using mapping
                        ambient_mass_fractions = {
                            species_mapping.get(species, species): value
                            for species, value in df_air_massfractions.iloc[0].items()
                            if species_mapping.get(species, species) in ct.Solution('gri30.yaml').species_names
                        }
                        new_total_mass_fraction = sum(ambient_mass_fractions.values())

                        # Normalize mass fractions so they sum to 1
                        if new_total_mass_fraction > 0:
                            normalized_air_mass_fractions = {
                                species: value / new_total_mass_fraction
                                for species, value in ambient_mass_fractions.items()
                            }
                        else:
                            normalized_air_mass_fractions = ambient_mass_fractions  # Avoid division by zero
                        # Saving raw and formated data
                        save_output_files_cantera(df_output, afterburning_filename, s_name)
                        # 03. Update df_engines with the results
                        # 03.1 Set all species mass fractions to zero for the current phase and engine pair
                        df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines.columns[df_engines.columns.get_loc('exit_temperature') + 1:]] = 0
                        # 03.2 Process df_output (which is a dictionary): Convert it to a DataFrame and drop the first 6 rows
                        df_output_filtered = pd.DataFrame.from_dict(df_output['Emissions'], orient='index', columns=['afterburning_emissions'])
                        
                        # 03.3 Transpose df_air_massfractions and adjust column names

                        df_air_massfractions = pd.DataFrame.from_dict(normalized_air_mass_fractions, orient='index', columns=['air_massfractions'])
                        
                        # 03.4 Combine the dataframes
                        combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)
                        
                        # Loop through the combined_df to calculate the emission index for each species
                        for species, row2 in combined_df.iterrows():
                            x_species = row2['afterburning_emissions']
                            x_species_atm = row2['air_massfractions']
                            EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)

                            # Update the species mass fraction in df_engines if the value is above the threshold
                            if EI_species > threshold:
                                df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), species] = EI_species     

                        # Fill NaN to avoid errors in the following calculations
                        df_engines = df_engines.fillna(0)
                    # Get mass fractions for the specific "phase and engine pair" from df_engines
                        # These are the precalculated data from the first Nasa CEA Calculation of the engines
                        # Alternativly its data from the Afterburning overwrite
                    df_mass_fractions = df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines.columns[df_engines.columns.get_loc('exit_temperature')+1:]] # The species start at exit_temperature + 1
                    
                    # Update Black Carbon Primary to Final Emission Index
                    if calculate_black_carbon:
                        
                        # Determine if the engine is a booster or stage engine
                        if engine == current_phase['booster_engine']:
                            current_fuel_type = current_phase['booster_fuel']
                        elif engine == current_phase['stage_engine']:
                            current_fuel_type = current_phase['stage_fuel']
                        else:
                            current_fuel_type = None
                        
                        # Transform the fuel type list into a single string with slashes
                        current_fuel_type_str = "/".join(current_fuel_type)
                        # Filter the dataframe to get the corresponding emission factors
                        df_emission_factor_fuel = df_emission_factors.loc[
                            (df_emission_factors['fuel'] == current_fuel_type_str) & 
                            (df_emission_factors['method'] == emission_factor_method)
                        ]

                        # Access the 'BC_prim' value, ensuring the filter has returned a row
                        if not df_emission_factor_fuel.empty:
                            primary_ei_bc = df_emission_factor_fuel.loc[:, 'BC_prim'].values[0]
                        else:
                            primary_ei_bc = 0.0
                            raise ValueError(f"No matching 'BC_prime' data found for fuel: {current_fuel_type_str} and method: {emission_factor_method}")
                        
                        final_ei_bc = black_carbon(row["h"], primary_ei_bc)
                        
                        # Include the BC Final Emission Index to df_mass_fractions
                        df_bc = pd.DataFrame({'BC': [final_ei_bc]}, index=df_mass_fractions.index)
                        df_mass_fractions = pd.concat([df_bc, df_mass_fractions], axis=1)

                    # Multiply mass fractions with interval_fuel_mass
                    df_species_mass = df_mass_fractions * interval_fuel_mass
                    df_species_mass = df_species_mass.fillna(0)
                    
                    # (d) Iterate through all species of current engine in iteration (c) and add them to the total_row_emissions DataFrame
                    for species in df_species_mass.columns:
                        if species in total_row_emissions.columns:
                            total_row_emissions.loc[idx, species] += df_species_mass[species].values[0]
                        else:
                            total_row_emissions.loc[idx, species] = df_species_mass[species].values[0]

            # Append total_row_emissions to total_emissions after iteration (b), (c) and (d) are finished
            total_emissions = pd.concat([total_emissions, total_row_emissions])
            total_emissions = total_emissions.fillna(0)
            total_emissions['total_sum_species'] = total_emissions.iloc[:, 10:].sum(axis=1)

        # Combine total_emissions into emission_results, while dropping the initial non-species columns of total_emissions
        emission_results = pd.concat([emission_results, total_emissions.iloc[:, 9:]], axis=1)
        
        
        # 4. Clean up - Remove thermo.lib and trans.lib
        clean_up()

    except Exception as e:
        data_processing.handle_error(e, "run_cantera", "Error in Cantera main function.")
    
    return emission_results
#endregion


#### NASA CEA - Main Function
def run_nasa_cea(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors, rocket_case):
    try:
        # 1. Inits 
        # Copy thermo.lib, trans.lib, cleaned ODE Results and create engine df
        init_directory()
        emission_results = df_trajectory_results_cleaned
        df_engines = create_engine_df(df_phases, df_launch_vehicles_engines)
        
        
        # 2. Rocket Problem - Engines
        # 2.1 Run NASA CEA Rocket Problem for every Engine (Booster or Stage)
        for idx, row in df_engines.iterrows():
            # Unique filenames for directory management
            filename = scenario_name + "_" + row["engine"]
            s_name = row["phase"] + "_rocket"
            problem = "rocket"

            # Creat .inp file, run NASA CEA & read results
            create_inp_file(problem, filename, rocket_case, df=row)
            subprocess.run(["echo", filename, "|", nasa_cea_folder_path + nasa_cea_exe], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            df_output = readCEA(problem, filename, rocket_case)
            
            # Add exit pressure and exit temperature to df_engines
            exit_pressure = df_output.loc['p', 'exit']
            exit_density = df_output.loc['rho', 'exit']
            exit_temperature = df_output.loc['t', 'exit']
            df_engines.loc[idx, 'exit_pressure'] = exit_pressure
            df_engines.loc[idx, 'exit_density'] = exit_density
            df_engines.loc[idx, 'exit_temperature'] = exit_temperature
            
            # Filter df_output and add species exit massfractions to df_engines
            df_species_filtered = df_output.iloc[6:, 2:].transpose()
            # Loop through each species in the filtered DataFrame
            for species in df_species_filtered.columns:
                # Get the value of the species
                value = df_species_filtered.loc["exit", species]
                
                # Check if the value is above the threshold
                if value > threshold:
                    # If the value is above the threshold, add it to df_engines
                    df_engines.loc[idx, species] = value
                else:
                    # If the value is below or equal to the threshold, remove the species column from df_engines
                    if species in df_engines.columns:
                        df_engines.loc[idx, species] = None
                        #df_engines.drop(columns=[species], inplace=True)

            # Saving raw and formated data
            save_files(filename, s_name, rocket_case)
            #clean_directory(filename)
            save_results_excel(filename, s_name, df_output, rocket_case)
        
        # 2.2 Save Results at Exit for all Engines to Excel and copy original data for subsequent Afterburning Calculations
        df_engines = df_engines.fillna(0)
        save_results_excel(scenario_name, "Rocket_Exit_Results", df_engines, rocket_case)
        df_engines_original_data = df_engines.copy()


        # 3. Total Emission Calculation
        # Initialize a DataFrame to hold all species emissions for all rows
        total_emissions = pd.DataFrame()

        # (a) Iterate through every "row" of the trajectory dataset
        for idx, row in emission_results.iterrows():
            
            # Filter data
            active_phases = row['active_phases']
            time_intervals = row['time_interval']
            # Initialize a DataFrame to hold all species emissions for the current row of iteration (a)
            total_row_emissions = pd.DataFrame(0, index=[idx], columns=emission_results.columns)


            # (b) Iterate through every "active phase and time interval pair" in the current row
                # Multiple phases can be active within one row of a compressed time interval
            for phase, interval in zip(active_phases, time_intervals):
                
                # Init/Filter data
                current_phase = df_phases.loc[phase]
                interval_start, interval_end = float(interval[0]), float(interval[1])
                # Time interval of the given data set (timestep or compressed time interval)
                time_interval = interval_end - interval_start
                # Time interval of the current phase in iteration (b)
                phase_time_interval = current_phase["time_end"] - current_phase["time_start"]
                # Function to create the current relevant Fuel Dataset (phase, engine, interval_fuel_mass)
                    # A list of tuples to store booster and stage data
                fuel_data = create_fuel_tuple(current_phase, phase, phase_time_interval, time_interval)
                
                
                # (c) Iterate through every "booster or stage" (in the new dataset fuel_data) for each "active phase and time interval pair" in the current row
                    # Multiple engines (booster or stage) can be active within the phases of iteration (b)
                for phase, engine, interval_fuel_mass in fuel_data:
                                                           
                    # Including Afterburning calculation if required
                    if afterburning:
                        # 01. Get required Data for NASA CEA Calculation
                        # 01.1 Athmospheric Data:
                            # at current position (from row, either average or last specified in config)
                        rho_atm, t_atm, p_atm, df_air_massfractions = data_processing.calculate_atmosphere_data(row)
                        dia = df_launch_vehicles['d'].iloc[0]
                        massflow, mdot, dmdot = trajectory.calculate_massflow(current_phase)
                        v_rocket = row["v"]
                        rof_ab = rof_ab_calc(rof_f, dia, massflow, rho_atm, v_rocket)
                              
                        # 01.2 Exhaust data:
                            # at exit from first engine rocket problem NASA CEA calculation (2.1)
                            # Access the original exit data of current "phase, engine pair" from df_engines_original_data
                        df_fuel_massfractions = df_engines_original_data.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines_original_data.columns[df_engines_original_data.columns.get_loc('exit_temperature')+1:]]
                                                
                        # Remove the unnecessary species from the dataset
                        for species in df_fuel_massfractions.columns:
                            # Get the value of the species
                            value = df_fuel_massfractions[species].values[0]
                            
                            # Check if the value is below or equal to the threshold
                            if value <= threshold:
                                # Drop the species column if the value is below or equal to the threshold
                                df_fuel_massfractions.drop(columns=[species], inplace=True)
                        
                        fuel_exit_temperature = df_engines_original_data.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), 'exit_temperature'].values[0]
                        fuel_exit_rho = df_engines_original_data.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), 'exit_density'].values[0]
                        
                        # 01.3 Problem Type Data:
                        t_ab = t_atm
                        p_ab = p_atm*10**(-5)
                        rho_ab = rho_atm
                        s_ab = None
                        t_fuel = fuel_exit_temperature
                        rho_fuel = fuel_exit_rho
                        t_air = t_atm
                        if t_air < 160.1:
                            t_air = 160.1

                        # 02. NASA CEA Afterburning Calculation of Exhaust and Atmosphere for current "phase, engine pair"
                        afterburning_filename = scenario_name + "_" + str(interval_start)
                        problem = "afterburning"
                        s_name = phase + "_" + engine + "_" + problem

                        # Creat .inp file, run NASA CEA & read results
                        create_inp_file(problem, afterburning_filename, None, None, t_ab, p_ab, rho_ab, s_ab, t_fuel, rho_fuel, t_air, rof_ab, df_fuel_massfractions, df_air_massfractions)
                        try:
                            subprocess.run(["echo", afterburning_filename, "|", nasa_cea_folder_path + nasa_cea_exe], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            df_output = readCEA(problem_afterburning, afterburning_filename)
                        
                            # Saving raw and formated data
                            save_files(afterburning_filename, s_name, rocket_case)
                            #clean_directory(afterburning_filename)
                            save_results_excel(afterburning_filename, s_name, df_output, rocket_case)
                        except Exception as e:
                            print(f"NASA CEA error: {e} at time {current_phase["time_start"]}s. Setting emissions fractions to 0")
                            df_output = pd.df_output({
                                'p': [0],
                                't': [0],
                                'mach': [0],
                                'H': [0],
                                'rho': [0],
                                's': [0],
                                'CO': [0]
                            })
                        
                        # 03. Update df_engines with the results
                        # 03.1 Set all species mass fractions to zero for the current phase and engine pair
                        df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines.columns[df_engines.columns.get_loc('exit_temperature') + 1:]] = 0
                        
                        # 03.2 Combine the dataframes df_output and df_air_massfractions to include all relevant species
                        # Drop the initial rows from df_output
                        df_output_filtered = df_output.iloc[6:]
                        df_output_filtered.columns = ['ab_output_filtered']
                        df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)
                        
                        # Transpose df_air_massfractions
                        df_air_massfractions = df_air_massfractions.transpose()
                        df_air_massfractions.columns = ['air_massfractions']
                        df_air_massfractions.index = df_air_massfractions.index.str.replace(r'^\*', '', regex=True)
                        
                        # Combine the dataframes
                        combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)
                                   
                        # Update the species mass fractions (emission factors) in df_engines for the current phase and engine pair
                        for species, row2 in combined_df.iterrows():
                            x_species = row2['ab_output_filtered']
                            x_species_atm = row2['air_massfractions']
                            EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)
                            
                            # Update the species mass fraction in df_engines if the value is above the threshold
                            if EI_species > threshold:
                                df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), species] = EI_species                        
                    
                    # Fill Nan to avoid errors in following calculation
                    df_engines = df_engines.fillna(0)
                    
                    # Get mass fractions for the specific "phase and engine pair" from df_engines
                        # These are the precalculated data from the first Nasa CEA Calculation of the engines
                        # Alternativly its data from the Afterburning overwrite
                    df_mass_fractions = df_engines.loc[(df_engines['phase'] == phase) & (df_engines['engine'] == engine), df_engines.columns[df_engines.columns.get_loc('exit_temperature')+1:]] # The species start at exit_temperature + 1

                    # Update Black Carbon Primary to Final Emission Index
                    if calculate_black_carbon:
                        
                        # Determine if the engine is a booster or stage engine
                        if engine == current_phase['booster_engine']:
                            current_fuel_type = current_phase['booster_fuel']
                        elif engine == current_phase['stage_engine']:
                            current_fuel_type = current_phase['stage_fuel']
                        else:
                            current_fuel_type = None
                        
                        # Transform the fuel type list into a single string with slashes
                        current_fuel_type_str = "/".join(current_fuel_type)

                        # Filter the dataframe to get the corresponding emission factors
                        df_emission_factor_fuel = df_emission_factors.loc[
                            (df_emission_factors['fuel'] == current_fuel_type_str) & 
                            (df_emission_factors['method'] == emission_factor_method)
                        ]
                        
                        # Access the 'BC_prim' value, ensuring the filter has returned a row
                        if not df_emission_factor_fuel.empty:
                            primary_ei_bc = df_emission_factor_fuel.loc[:, 'BC_prim'].values[0]
                        else:
                            primary_ei_bc = 0.0
                            raise ValueError(f"No matching 'BC_prime' data found for fuel: {current_fuel_type_str} and method: {emission_factor_method}")
                        
                        final_ei_bc = black_carbon(row["h"], primary_ei_bc)
                        
                        # Include the BC Final Emission Index to df_mass_fractions
                        df_bc = pd.DataFrame({'BC': [final_ei_bc]}, index=df_mass_fractions.index)
                        df_mass_fractions = pd.concat([df_bc, df_mass_fractions], axis=1)

                    # Multiply mass fractions with interval_fuel_mass
                    df_species_mass = df_mass_fractions * interval_fuel_mass
                    df_species_mass = df_species_mass.fillna(0)
                    
                                                      
                    # (d) Iterate through all species of current engine in iteration (c) and add them to the total_row_emissions DataFrame
                    for species in df_species_mass.columns:
                        if species in total_row_emissions.columns:
                            total_row_emissions.loc[idx, species] += df_species_mass[species].values[0]
                        else:
                            total_row_emissions.loc[idx, species] = df_species_mass[species].values[0]

            # Append total_row_emissions to total_emissions after iteration (b), (c) and (d) are finished
            total_emissions = pd.concat([total_emissions, total_row_emissions])
            total_emissions = total_emissions.fillna(0)
            total_emissions['total_sum_species'] = total_emissions.iloc[:, 10:].sum(axis=1)

        # Combine total_emissions into emission_results, while dropping the initial non-species columns of total_emissions
        emission_results = pd.concat([emission_results, total_emissions.iloc[:, 9:]], axis=1)
        
        
        # 4. Clean up - Remove thermo.lib and trans.lib
        clean_up()
        
    except Exception as e:
        data_processing.handle_error(e, "run_nasa_cea", "Error in NASA CEA main function.")
    
    return emission_results
#endregion
