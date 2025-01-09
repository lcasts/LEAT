### This file includes all functions to process the data

### Imports
from input_data.config import *
import os
import pandas as pd
from pymsis import msis
from ast import literal_eval

#region: General Data Handling Functions
# Import all required data based on user input and config
def get_input_data():
    # 0. Additional function for data cleaning
    def clean_launchvehicle_dataframe(df_launch_vehicles):
        # Data columns to slit including "/" separator
        columns_to_split = ['booster_fuel', 'booster_fuel_temp', 'booster_propellant_massfractions', 
                            'stage1_fuel', 'stage1_fuel_temp', 'stage1_propellant_massfractions', 
                            'stage2_fuel', 'stage2_fuel_temp', 'stage2_propellant_massfractions', 
                            'stage3_fuel', 'stage3_fuel_temp', 'stage3_propellant_massfractions', 
                            'stage4_fuel', 'stage4_fuel_temp', 'stage4_propellant_massfractions']
        
        # Function to split columns
        def split_data(data):
            if pd.isna(data):
                return []  # Return an empty list for NaN values
            else:
                try:
                    return str(data).split('/')  # Convert to string and then split
                except Exception:
                    return []  # Return an empty list for other exceptions
        
        try:
            # Split columns
            for column in columns_to_split:
                df_launch_vehicles[column] = df_launch_vehicles[column].apply(split_data)
                
            # Drop last columns
            source_index = df_launch_vehicles.columns.get_loc("Source")
            df_launch_vehicles = df_launch_vehicles.iloc[:, :source_index]
            
            return df_launch_vehicles
        
        except Exception as e:
            handle_error(e, "clean_launchvehicle_dataframe", "LV Dataframe could not be cleaned")
    
    # 1. Import and Clean Data 
    try:
        # 1) Import Emission Factors
        df_emission_factors = pd.DataFrame()
        if use_emission_factors or (use_nasa_cea and calculate_black_carbon):
            ef_path = os.path.join(input_data_folder_path, file_name_emission_factors)
            df_emission_factors = pd.read_excel(ef_path, index_col=None, sheet_name=sheet_name_emission_factors)
            df_emission_factors = df_emission_factors.fillna(0)
            if 'notes' in df_emission_factors.columns:
                # Select columns up to (but not including) "notes"
                df_emission_factors = df_emission_factors.loc[:, : "notes"].iloc[:, :-1]
        
        # 2) Import Launch Rates
        # df_launch_rates = pd.DataFrame()
        # if launch_rates:
        #     lr_path = os.path.join(input_data_folder_path, file_name_launch_rates)
        #     df_launch_rates = pd.read_excel(lr_path, index_col=0, sheet_name=sheet_name_launch_rates)
        
        # 3) Import Launch Sites
        ls_path = os.path.join(input_data_folder_path, file_name_launch_sites)
        df_launch_sites = pd.read_excel(ls_path, index_col=0, sheet_name=sheet_name_launch_sites)
        
        # 4) Scenarios
        # 4.1) Import Scenarios
        s_path = os.path.join(input_data_folder_path, file_name_scenarios)
        df_scenarios = pd.read_excel(s_path, index_col=0, sheet_name=sheet_name_scenarios)
        df_scenarios.drop('note', axis=1, inplace=True)
        # 4.2) Filter Scenarios
        if all_scenarios:
            pass
        else:
            df_scenarios = df_scenarios.loc[df_scenarios.index.isin(user_defined_scenarios)]
        # 4.3) Include Launch Sites Data
        df_scenarios = pd.merge(df_scenarios, df_launch_sites[['lat', 'lon', 'alt']], left_on='launch_site', right_index=True)
        
        # 5) Launch Vehicles
        # 5.1) Import Launch Vehicles with Filter
        lv_path = os.path.join(input_data_folder_path, file_name_launch_vehicle)
        df_launch_vehicles = pd.read_excel(lv_path, index_col=0, sheet_name=sheet_name_launch_vehicle)
        # 5.2) Filter Launch Vehicles
        launch_vehicles_to_filter = df_scenarios['launch_vehicle'].unique().tolist()
        df_launch_vehicles = df_launch_vehicles.loc[df_launch_vehicles.index.isin(launch_vehicles_to_filter)]
        # 5.3) Data Cleaning of Dataframe
        df_launch_vehicles = clean_launchvehicle_dataframe(df_launch_vehicles)
        
        # 6) Engines
        # 6.1) Import Engines
        df_launch_vehicles_engines = pd.read_excel(lv_path, index_col=0, sheet_name=sheet_name_launch_vehicle_engines)
        df_launch_vehicles_engines.drop('note', axis=1, inplace=True)
        # 6.2) Filter Engines
        unique_engines_to_filter = pd.unique(df_launch_vehicles[['booster_engine', 'stage1_engine', 'stage2_engine', 'stage3_engine', 'stage4_engine']].values.flatten()).tolist()
        df_launch_vehicles_engines = df_launch_vehicles_engines.loc[df_launch_vehicles_engines.index.isin(unique_engines_to_filter)]

        return df_emission_factors, df_scenarios, df_launch_sites, df_launch_vehicles, df_launch_vehicles_engines

    except Exception as e:
        handle_error(e, "get_input_data", "Input data could not be processed")     

# Initial Console Prints
def print_scenario_data(idx, scenario_name, df_scenarios):
    try:
        print("\n" + colors.BOLD + colors.YELLOW + "Scenario #{}:".format(idx), scenario_name + colors.END)
        print(colors.BOLD + colors.BLUE + "Scenario Data:" + colors.END)
        print(df_scenarios.loc[scenario_name].to_string())
    except Exception as e:
        handle_error(e, "print_scenario_data", "Error while locating scenario data.")

# File naming handling
def get_unique_filename(base_name, folder, type):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    counter = 1
    filename = f"{base_name}{type}"
    file_path = os.path.join(folder, filename)

    while os.path.exists(file_path):
        filename = f"{base_name}({counter}){type}"
        file_path = os.path.join(folder, filename)
        counter += 1
    
    return filename, file_path

# General Error Handling Function
def handle_error(error, function_name, function_message):
    error_messages = {
        FileNotFoundError: "One or more input files not found. Please check the file paths.",
        pd.errors.EmptyDataError: "One or more input files are empty. Please ensure they contain data.",
        pd.errors.ParserError: "Error parsing Excel file. Please ensure the file format is correct.",
        ValueError: "Please check your input files and configurations.",
        Exception: "An unexpected error occurred. Please check your input files and configurations."
    }

    if type(error) in error_messages:
        print(colors.BOLD + colors.RED + f"Error in function '{function_name}': {error}")
        print(f"{function_message}: " + error_messages[type(error)] + colors.END)
    else:
        print(colors.BOLD + colors.RED + f"An unexpected error occurred in function '{function_name}': {error}")
        print(f"{function_message}: Please check your input files and configurations." + colors.END)

    return

# Load External Trajectory Data
def load_external_trajectory_data():
    # Load external trajectory data from CSV file specified in filename_ext_traj
    df_trajectory_results_cleaned = pd.read_csv(file_name_ext_trajectory)
    df_trajectory_results_cleaned['active_phases'] = df_trajectory_results_cleaned['active_phases'].apply(
        lambda x: [] if x == '[]' else literal_eval(x) if isinstance(x, str) else x
    )
    df_trajectory_results_cleaned['time_interval'] = df_trajectory_results_cleaned['time_interval'].apply(
        lambda x: [] if x == '[]' else literal_eval(x) if isinstance(x, str) else x
    )
    print(colors.BOLD + colors.GREEN + f"Successfully loaded trajectory results from external file: '{file_name_ext_trajectory}'" + colors.END)
    
    return df_trajectory_results_cleaned
#endregion


#region: Data Processing for Trajectory ODE
# Calculating Phases Data based on user input
def phases(scenario_name, df_scenarios, df_launch_vehicles):
    phases = pd.DataFrame(columns=['time_start', 'time_end', 'mass_start', 'mass_burnout', 'mass_separation', 'booster_engine', 'booster_force_sl', 'booster_force_vac', 'booster_fuel', 'booster_fuel_temp', 'booster_fuel_mass', 'stage_engine', 'stage_force_sl', 'stage_force_vac', 'stage_fuel', 'stage_fuel_temp', 'stage_fuel_mass'])
    phase_num = 1   # Counter
    time_start = 0  # Start time default
    
    try:
        # 1. Filter df_launch_vehicles based on scenario
        launch_vehicle = df_scenarios.loc[scenario_name]['launch_vehicle']
        df_launch_vehicle_scenario = df_launch_vehicles[df_launch_vehicles.index == launch_vehicle]
        
        # 2. Calculate total mass and massflow
        mass_columns_to_sum = ['booster_total_propellant', 'booster_dry_mass',
                               'stage1_total_propellant', 'stage1_dry_mass',
                               'stage2_total_propellant', 'stage2_dry_mass',
                               'stage3_total_propellant', 'stage3_dry_mass',
                               'stage4_total_propellant', 'stage4_dry_mass']
        mass_payload = df_scenarios.loc[scenario_name]['mass_payload']
        mass_total = df_launch_vehicle_scenario[mass_columns_to_sum].sum(axis=1)
        mass_start = mass_total[launch_vehicle] + mass_payload
        stage1_massflow = df_launch_vehicle_scenario['stage1_total_propellant'].iloc[0] / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
        
        # 3. Loop through each phase based on the fuel columns as reference for available data
        for col in ['booster_fuel', 'stage1_fuel', 'stage2_fuel', 'stage3_fuel', 'stage4_fuel']:
            # Check if the fuel array is empty
            if not df_launch_vehicle_scenario[col].any():
                break
            
            # Check if stage 1 starts with the booster
            start_with_booster = df_launch_vehicle_scenario['start_with_booster'].iloc[0]
            
            # 3.1) Calculate end time and burnout mass for current phase
            # Phase 1: Only Booster
            if phase_num == 1 and not start_with_booster:
                time_end = df_launch_vehicle_scenario['booster_burntime'].iloc[0]
                mass_burnout = mass_start - df_launch_vehicle_scenario[col.replace("_fuel", "_total_propellant")].iloc[0]
            # Phase 1: Booster & Stage 1
            if phase_num == 1 and start_with_booster:
                time_end = df_launch_vehicle_scenario['booster_burntime'].iloc[0]
                fuel_mass_stage1 = stage1_massflow * time_end
                mass_burnout = mass_start - df_launch_vehicle_scenario[col.replace("_fuel", "_total_propellant")].iloc[0] - fuel_mass_stage1
            
            # Phase 2: Stage 1 (Start after Booster)
            elif phase_num == 2 and not start_with_booster:
                time_end = time_start + df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                delta_t = time_end - time_start
                fuel_mass_stage1 = stage1_massflow * delta_t
                mass_burnout = mass_start - fuel_mass_stage1
            # Phase 2: Stage 1 (Started with Booster)
            elif phase_num == 2 and start_with_booster:
                time_end = df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                delta_t = time_end - time_start
                fuel_mass_stage1 = stage1_massflow * delta_t
                mass_burnout = mass_start - fuel_mass_stage1
            
            # Phase 3+: Stage >= 2
            else:
                time_end = time_start + df_launch_vehicle_scenario[col.replace("_fuel", "_burntime")].iloc[0]
                mass_burnout = mass_start - df_launch_vehicle_scenario[col.replace("_fuel", "_total_propellant")].iloc[0]
                delta_t = time_end - time_start
            
            # 3.2) Calculate separation mass for current phase
            mass_separation = mass_burnout - df_launch_vehicle_scenario[col.replace("_fuel", "_dry_mass")].iloc[0]
            
            # 3.3) General data of Booster and Stage according to phase
            if phase_num == 1:
                booster_engine = df_launch_vehicle_scenario["booster_engine"].iloc[0]
                booster_force_sl = df_launch_vehicle_scenario["booster_force_sl"].iloc[0]
                booster_force_vac = df_launch_vehicle_scenario["booster_force_vac"].iloc[0]
                booster_fuel = df_launch_vehicle_scenario["booster_fuel"].iloc[0]
                booster_fuel_temp = df_launch_vehicle_scenario["booster_fuel_temp"].iloc[0]
                
                if start_with_booster:
                    stage_engine = df_launch_vehicle_scenario["stage1_engine"].iloc[0]
                    stage_force_sl = df_launch_vehicle_scenario["stage1_force_sl"].iloc[0]
                    stage_force_vac = df_launch_vehicle_scenario["stage1_force_vac"].iloc[0]
                    stage_fuel = df_launch_vehicle_scenario["stage1_fuel"].iloc[0]
                    stage_fuel_temp = df_launch_vehicle_scenario["stage1_fuel_temp"].iloc[0]
                else:
                    stage_engine = np.nan
                    stage_force_sl = np.nan
                    stage_force_vac = np.nan
                    stage_fuel = np.nan
                    stage_fuel_temp = np.nan
                                    
            else:
                booster_engine = np.nan
                booster_force_sl = np.nan
                booster_force_vac = np.nan
                booster_fuel = np.nan
                booster_fuel_temp = np.nan
                booster_fuel_mass = np.nan
                              
                stage_engine = df_launch_vehicle_scenario[col.replace("_fuel", "_engine")].iloc[0]
                stage_force_sl = df_launch_vehicle_scenario[col.replace("_fuel", "_force_sl")].iloc[0]
                stage_force_vac = df_launch_vehicle_scenario[col.replace("_fuel", "_force_vac")].iloc[0]
                stage_fuel = df_launch_vehicle_scenario[col].iloc[0]    # No replace, its variable in loop           
                stage_fuel_temp = df_launch_vehicle_scenario[col.replace("_fuel", "_fuel_temp")].iloc[0]

            # 3.4) Calculate propellant masses
            total_propellant_data = df_launch_vehicle_scenario[col.replace("_fuel", "_total_propellant")].iloc[0]
            massfractions_data = df_launch_vehicle_scenario[col.replace("_fuel", "_propellant_massfractions")].iloc[0]
            mass_fuel_data = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_fuel")].iloc[0]
            mass_ox_data = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_ox")].iloc[0]
            
            booster_fuel_mass = np.nan
            stage_fuel_mass = np.nan
            
            # Check conditions (Massfraction data or ROF data) and perform actions accordingly
            if not massfractions_data:  # If massfractions_data is empty
                try:
                    if phase_num == 1:
                        booster_mass_fuel = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_fuel")].iloc[0]
                        booster_mass_ox = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_ox")].iloc[0]
                        booster_fuel_mass = [booster_mass_fuel, booster_mass_ox]
                    
                        if start_with_booster:
                            stage_fuel_mass = []
                            if not df_launch_vehicle_scenario['stage1_propellant_massfractions'].iloc[0]:
                                stage_fuel_massflow = df_launch_vehicle_scenario['stage1_mass_fuel'].iloc[0] / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                stage_mass_fuel = stage_fuel_massflow * time_end
                                stage_ox_massflow = df_launch_vehicle_scenario['stage1_mass_ox'].iloc[0] / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                stage_mass_ox = stage_ox_massflow * time_end
                                stage_fuel_mass = [stage_mass_fuel, stage_mass_ox]
                                
                            else:
                                for massfraction in df_launch_vehicle_scenario['stage1_propellant_massfractions'].iloc[0]:
                                    constituent_mass = float(massfraction) * df_launch_vehicle_scenario['stage1_total_propellant'].iloc[0]
                                    constituent_massflow  = constituent_mass / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                    stage_fuel_mass.append(constituent_massflow * time_end)
                        
                    else:   # Phase > 1
                        stage_fuel_massflow = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_fuel")].iloc[0] / df_launch_vehicle_scenario[col.replace("_fuel", "_burntime")].iloc[0]
                        stage_mass_fuel = stage_fuel_massflow * delta_t
                    
                        stage_ox_massflow = df_launch_vehicle_scenario[col.replace("_fuel", "_mass_ox")].iloc[0] / df_launch_vehicle_scenario[col.replace("_fuel", "_burntime")].iloc[0]
                        stage_mass_ox = stage_ox_massflow * delta_t
                        
                        stage_fuel_mass = [stage_mass_fuel, stage_mass_ox]
                except Exception as e:
                    handle_error(e, "phases", "Error in fuel mass calculation using oxidizer and fuel data.")
            
            else:  # If massfractions_data
                try:
                    if phase_num == 1:
                        booster_fuel_mass = []
                        for massfraction in massfractions_data:
                            constituent_mass = float(massfraction) * total_propellant_data
                            booster_fuel_mass.append(constituent_mass)
                            
                        if start_with_booster:
                            stage_fuel_mass = []
                            
                            if not df_launch_vehicle_scenario['stage1_propellant_massfractions'].iloc[0]:
                                stage_fuel_massflow = df_launch_vehicle_scenario['stage1_mass_fuel'].iloc[0] / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                stage_mass_fuel = stage_fuel_massflow * time_end
                                stage_ox_massflow = df_launch_vehicle_scenario['stage1_mass_ox'].iloc[0] / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                stage_mass_ox = stage_ox_massflow * time_end
                                stage_fuel_mass = [stage_mass_fuel, stage_mass_ox]
                                
                            else:
                                for massfraction in df_launch_vehicle_scenario['stage1_propellant_massfractions'].iloc[0]:
                                    constituent_mass = float(massfraction) * df_launch_vehicle_scenario['stage1_total_propellant'].iloc[0]
                                    constituent_massflow  = constituent_mass / df_launch_vehicle_scenario['stage1_burntime'].iloc[0]
                                    stage_fuel_mass.append(constituent_massflow * time_end)
                    
                    else:   # Phase > 1
                        stage_fuel_mass = []
                        for massfraction in massfractions_data:
                            constituent_mass = float(massfraction) * total_propellant_data
                            constituent_massflow  = constituent_mass / df_launch_vehicle_scenario[col.replace("_fuel", "_burntime")].iloc[0]
                            stage_fuel_mass.append(constituent_massflow * delta_t)
                                                    
                except Exception as e:
                    handle_error(e, "phases", "Error in fuel mass calculation using massfractions.")                
            
            # 3.5) Create a new row for current phase in the loop
            phases.loc["Phase_" + str(phase_num)] = [time_start, time_end, mass_start, mass_burnout, mass_separation, booster_engine, booster_force_sl, booster_force_vac, booster_fuel, booster_fuel_temp, booster_fuel_mass, stage_engine, stage_force_sl, stage_force_vac, stage_fuel, stage_fuel_temp, stage_fuel_mass]
            
            # 3.5) Update data for next phase
            phase_num += 1
            time_start = time_end
            mass_start = mass_separation
        
        print(colors.BOLD + colors.BLUE + "Phases Data (Time & Mass):\n" + colors.END + phases.iloc[:, :5].to_string())
        print(colors.BOLD + colors.BLUE + "Phases Data (Booster):\n" + colors.END + phases.iloc[:, 5:11].to_string())
        print(colors.BOLD + colors.BLUE + "Phases Data (Stage):\n" + colors.END + phases.iloc[:, 11:].to_string())
        
    # Error handling
    except Exception as e:
        handle_error(e, "phases", "Phases could not be calculated")

    return phases

# ODE Arguments and Initial Values
def ode_args_and_initial_values(scenario_name, df_scenarios, df_launch_vehicles, phases):
    try:
        scenario = df_scenarios.loc[scenario_name]
        launch_vehicle = df_launch_vehicles.loc[scenario['launch_vehicle']]
        h_0 = scenario['alt']
        m_0 = phases.loc['Phase_1', 'mass_start']
        date_0 = np.datetime64(scenario['date']).astype('datetime64[s]').astype(int)
        lat_0 = scenario['lat']
        lon_0 = scenario['lon']
        phase_intervals = [phases.at[f'Phase_{i}', 'time_start'] for i in range(1, len(phases) + 1)]
        phase_intervals.append(phases.at[f'Phase_{len(phases)}', 'time_end'])
        atmos_0 = msis.run(scenario["date"], lon_0, lat_0, 0)
        rho_0 = atmos_0[0, 0]
        temp_0 = atmos_0[0, 10]
        p_0 = rho_0 * temp_0 * R
        F_0 = 0
        massflow_0 = 0
        
        ode_args = [g_0, R_0, p_0, max_step, scenario, launch_vehicle, phases, phase_intervals]
        y_0 = np.array([v_0, x_0, s_0, h_0, gamma_0, m_0, date_0 , lat_0, lon_0, p_0, rho_0, temp_0, F_0, massflow_0])
        
    except Exception as e:
        handle_error(e, "ode_args_and_initial_values", "Error finding required data.")
    
    return ode_args, y_0

# Post data processing of ODE results
def process_and_save_ODE_results(results, scenario_name):    
    output_data = pd.DataFrame({
        "t": results.t,
        "h": results.y[3] / 1000,
        "x": results.y[1] / 1000,
        "s": results.y[2] / 1000,
        "m": results.y[5],
        "gamma": results.y[4],
        "v": results.y[0],
        "date": results.y[6],
        "lon": results.y[8],
        "lat": results.y[7],
        "p": results.y[9],
        "rho": results.y[10],
        "temp": results.y[11],
        "F": results.y[12],
        "d_m": results.y[13]    # Only massflow of fuel per timestep, not separation mass
    })
    
    # Convert Unix timestamp to datetime format
    #output_data['date'] = pd.to_datetime(output_data['date'], unit='s')
    output_data['date'] = pd.to_datetime(output_data['date'], unit='s').to_numpy(dtype='datetime64[s]')

    # Saving data to csv file
    try:
        base_name = f"{output_data_trajectory_name}{output_data_raw_name}{scenario_name}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        
        output_data.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written trajectory results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Trajectory Data:" + colors.END)
        print(output_data.head())
        
    except Exception as e:
        handle_error(e, "process_and_save_ODE_results", "Error finding required data.")
    
    return output_data

# Compress the Trajectory Data based on in config specified method: by time or by height
def compress_and_save_trajectory_results(output_data, scenario_name):
    # Compress Data
    try:
        # Add First Row: Init dataframe with the first row
        compressed_df = pd.DataFrame(output_data.iloc[[0]])
        compressed_df['d_m'] = 0
        
        # Compress based on time
        if compress_method == "time":
            current_time = compress_interval
            d_m_sum = 0
            
            if compress_atmosphere == "latest":
                # Loop through "output_data" based on time intervals
                for idx in range(1, len(output_data)):
                    if output_data['t'].iloc[idx] < current_time + interval_tolerance:
                        # Sum up the d_m value for this row
                        d_m_sum += output_data['d_m'].iloc[idx]
                    
                    if output_data['t'].iloc[idx] >= current_time - interval_tolerance:
                        row = output_data.iloc[[idx]].copy()
                        row['d_m'] = d_m_sum
                        compressed_df = pd.concat([compressed_df, row], ignore_index=True)
                        d_m_sum = 0
                        current_time += compress_interval
                        
            elif compress_atmosphere == "averages":
                interval_data = []

                # Loop through "output_data" based on time intervals
                for idx in range(1, len(output_data)):
                    interval_data.append(output_data.iloc[idx])
                    
                    if output_data['t'].iloc[idx] < current_time + interval_tolerance:
                        # Sum up the d_m value for this row
                        d_m_sum += output_data['d_m'].iloc[idx]
                        
                    # Calculate the average of atmospheric data when interval length is set 
                    if output_data['t'].iloc[idx] >= current_time - interval_tolerance:
                        interval_df = pd.DataFrame(interval_data)
                        interval_avg = interval_df[['p', 'rho', 'temp']].mean()
                        averaged_row = output_data.iloc[[idx]].copy()
                        averaged_row.loc[:, 'p'] = interval_avg['p']
                        averaged_row.loc[:, 'rho'] = interval_avg['rho']
                        averaged_row.loc[:, 'temp'] = interval_avg['temp']

                        averaged_row['d_m'] = d_m_sum  # Overwrite the d_m value with the accumulated sum

                        # Add the updated averaged row to the dataframe and repeat for next interval
                        compressed_df = pd.concat([compressed_df, averaged_row], ignore_index=True)
                        d_m_sum = 0
                        current_time += compress_interval
                        interval_data = []

        # Compress based on height
        elif compress_method == "height":
            last_height = output_data['h'].iloc[-1]
            height = 0
            d_m_sum = 0

            # Hol den ersten Eintrag der Höhe (erste Zeile von height_data)
            first_height_value = output_data['h'].iloc[0]

            # Berechne das abgerundete ganzzahlige Vielfache n
            if first_height_value > height+compress_interval:
                # Berechne das Vielfache n als abgerundeten Wert
                n = int((first_height_value - height) // compress_interval) 
                # Setze die Starthöhe auf height + compress_interval * n
                height = height + compress_interval * n    
                    
            if compress_atmosphere == "latest":
                # Loop through "output_data" based on height intervals
                while height <= last_height:
                    # Find data rows corresponding to current height interval
                    height_data = output_data.query(f'h >= {height} and h < {height + compress_interval}')
                     
                    if not height_data.empty:
                        # Sum the d_m values for the interval
                        for idx in range(height_data.shape[0]):
                            d_m_sum += height_data['d_m'].iloc[idx]  
                                          
                        # Append the first row of the interval if it exists
                        row = height_data.iloc[[-1]]  # Use the last row of the height interval
                        row['d_m'] = d_m_sum  # Set the summed d_m value for the interval
                        compressed_df = pd.concat([compressed_df, row], ignore_index=True)
                    
                    # Move to the next interval
                    height += compress_interval
                    d_m_sum = 0
                
            elif compress_atmosphere == "averages":
                interval_data = []

                # Loop through "output_data" based on height intervals
                while height <= last_height:
                    # Adding all rows within the interval
                    height_data = output_data.query(f'h >= {height} and h < {height + compress_interval}')
                    interval_data.append(height_data)
                    
                    if not height_data.empty:
                        # Sum the d_m values for the interval
                        for idx in range(height_data.shape[0]):
                            d_m_sum += height_data['d_m'].iloc[idx]
                        
                        # Calculate the average of atmospheric data within the interval
                        interval_df = pd.concat(interval_data, ignore_index=True)
                        interval_avg = interval_df[['p', 'rho', 'temp']].mean()
                        averaged_row = height_data.iloc[[-1]].copy()
                        averaged_row.loc[:, 'p'] = interval_avg['p']
                        averaged_row.loc[:, 'rho'] = interval_avg['rho']
                        averaged_row.loc[:, 'temp'] = interval_avg['temp']
                        
                        averaged_row['d_m'] = d_m_sum  # Overwrite d_m value with sum
                        
                        # Add the updated averaged row to the dataframe and repeat for next interval
                        compressed_df = pd.concat([compressed_df, averaged_row], ignore_index=True)
                        height += compress_interval
                        interval_data = []
                        d_m_sum = 0


    except Exception as e:
        handle_error(e, "compress_and_save_trajectory_results", "Error compressing the data.")
    
    # Saving Compressed Data to csv file
    try:
        base_name = f"{output_data_trajectory_name}{output_data_compression_name}{scenario_name}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        
        compressed_df.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written compressed trajectory results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Compressed Trajectory Data:" + colors.END)
        print(compressed_df.head())
        
    except Exception as e:
        handle_error(e, "compress_and_save_trajectory_results", "Error finding required data.")    
    
    return compressed_df
#endregion


#region: Data Processing for Emission Calculations
# Preprocessing Function for Emission Calculation with compressed/uncompressed ODE results
def emission_preprocessing_data(df_trajectory_results, df_phases):
    try:
        # Filter ODE results and add columns
        df_cleaned = df_trajectory_results.loc[:, ["t", "h", "date", "lon", "lat", "v", "d_m",]]
        df_cleaned["active_phases"] = None
        df_cleaned["time_interval"] = None
        
        # Filter time data from phases - Convert the phase information to a list of tuples
        phase_times = []
        for idx, row in df_phases.iterrows():
            start_time = row['time_start']
            end_time = row['time_end']
            phase_times.append((idx, start_time, end_time))
                
        # Iterate through filtered dataframe and include active phases and their time intervals
        for idx, row in df_cleaned.iterrows():
            if idx == 0:
                df_cleaned.at[idx, "time_interval"] = []
                df_cleaned.at[idx, "active_phases"] = []
                last_time = row['t']
            else:
                current_time = row['t']
                active_phases = []
                time_intervals = []

                for phase_name, start_time, end_time in phase_times:
                    if last_time < end_time and current_time > start_time:
                        interval_start = round(max(last_time, start_time), 7)
                        interval_end = round(min(current_time, end_time), 7)
                        active_phases.append(phase_name)
                        time_intervals.append([interval_start, interval_end])

                df_cleaned.at[idx, "active_phases"] = active_phases
                df_cleaned.at[idx, "time_interval"] = time_intervals
                last_time = current_time    

    except Exception as e:
        handle_error(e, "emission_preprocessing_data", "Error while preprocessing data for Emission Calculation.")
        
    return df_cleaned        

# Saving the Results of the Emission Calculations
def save_emission_results(emission_results, scenario_name, type):
    # Saving data to csv file
    try:
        if emission_results is None or not isinstance(emission_results, pd.DataFrame):
            raise ValueError("Emission results are not valid DataFrame")
        
        # Clean column names
        emission_results.columns = [col.replace("*", "") for col in emission_results.columns]
                
        # Saving data to csv file
        base_name = f"{output_data_emissions_name}{scenario_name}_{type}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_nasa_cea, ".csv")
        
        emission_results.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written emission results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + f"Emission Data {type}:" + colors.END)
        print(emission_results.head())
        
    except Exception as e:
        handle_error(e, "save_emission_results", "Error finding required data.")

# Calculate Athmosphere Data for Afterburning
def calculate_atmosphere_data(row):
    # Helper function to handle NaN values
    def handle_nan(value):
        return 0 if np.isnan(value) else value
    
    # Extract date, longitude, latitude, and height from the row
    date = row['date']
    lon = row['lon']
    lat = row['lat']
    height = row['h']
    
    # Run msis Atmosphere Model Function
    atmosphere = msis.run(date, lon, lat, height)
    
    # Total mass density (kg/m3)
    rho_atm = handle_nan(atmosphere[0, 0])
    t_atm = handle_nan(atmosphere[0, 10])
    p_atm = rho_atm * t_atm * R
    
    # Raw Density Data of Species (in molecules per m3)
    densities = {
        '*N2': handle_nan(atmosphere[0, 1]),
        '*O2': handle_nan(atmosphere[0, 2]),
        '*O': handle_nan(atmosphere[0, 3]),
        '*He': handle_nan(atmosphere[0, 4]),
        '*H': handle_nan(atmosphere[0, 5]),
        '*Ar': handle_nan(atmosphere[0, 6]),
        'N': handle_nan(atmosphere[0, 7]),
        #'aox': handle_nan(atmosphere[0, 8]),
        '*NO': handle_nan(atmosphere[0, 9])
    }
    
    # Convert number densities to mass densities (kg/m3)
    mass_densities = {species: densities[species] * molar_masses[species] / avogadro_number for species in densities}
    
    # Calculate mass fractions (kg/kg)
    mass_fractions = {species: mass_densities[species] / rho_atm for species in mass_densities}

    # Convert mass fractions dictionary to DataFrame
    df_atm_massfractions = pd.DataFrame([mass_fractions])
    
    return rho_atm, t_atm, p_atm, df_atm_massfractions
    
#endregion