# 0. Import Libraries and Functions
from input_data.config import *
from scripts import emissions
from scripts import trajectory
from scripts import data_processing

# 1) Import Data and Initial Data Cleaning
df_emission_factors, df_scenarios, df_launch_sites, df_launch_vehicles, df_launch_vehicles_engines = data_processing.get_input_data()

# 2) Iterating through all specified scenarios to process
for idx, scenario_name in enumerate(df_scenarios.index, start=1):
    data_processing.print_scenario_data(idx, scenario_name, df_scenarios)
    df_phases = data_processing.phases(scenario_name, df_scenarios, df_launch_vehicles)
    
    # 2.0) Load External Trajectory Data if specified
    if use_external_trajectory_data:
        df_trajectory_results_cleaned = data_processing.load_external_trajectory_data()

    # 2.1) Calculating Trajectory if not using External Trajectory Data
    if not use_external_trajectory_data:
        # a) Preprocessing Data for ODE
        ode_args, y_0 = data_processing.ode_args_and_initial_values(scenario_name, df_scenarios, df_launch_vehicles, df_phases)

        # b) Solving Trajectory ODE
        raw_trajectory_results = trajectory.solve_ODE(y_0, ode_args)

        # c) Saving ODE Results
        df_trajectory_results = data_processing.process_and_save_ODE_results(raw_trajectory_results, scenario_name) # Raw Data
        if compress_data:
            df_trajectory_results = data_processing.compress_and_save_trajectory_results(df_trajectory_results, scenario_name) # Compressed Data    

        # d) Preprocess the Trajectory ODE Results for Emission Calculation
        df_trajectory_results_cleaned = data_processing.emission_preprocessing_data(df_trajectory_results, df_phases)
        df_trajectory_results_cleaned.to_csv(f"output_data/trajectory/trajectory_{scenario_name}_input.csv", index=False) 
        
    # 2.2) Calculating Emissions with Emission Factors or NASA CEA
    if calculate_emissions:
        # a) Calculate Emissions
        if use_emission_factors:
            emission_results = emissions.emission_factors(df_trajectory_results_cleaned, df_phases, df_emission_factors, scenario_name)
            data_processing.save_emission_results(emission_results, scenario_name, "Emission_Factors")
            
        if use_nasa_cea:
            if rocket_equilibrium:
                rocket_case = "equilibrium"
                emission_results = emissions.run_nasa_cea(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors, rocket_case)
                data_processing.save_emission_results(emission_results, scenario_name, "NASA_CEA_Equilibrium")  
            if rocket_frozen:
                rocket_case = "frozen"
                emission_results = emissions.run_nasa_cea(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors, rocket_case)
                data_processing.save_emission_results(emission_results, scenario_name, "NASA_CEA_Frozen")
        
        if use_cantera:   
            emission_results = emissions.run_cantera(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors)
            data_processing.save_emission_results(emission_results, scenario_name, "Cantera")