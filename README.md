# **LEAT**

### _Launch Emission Assessment Tool_

LEAT is a Python-based tool licensed under the GNU Affero General Public License v3.0. The licence can be found in LICENCE.md. It is designed to calculate the global emissions of specified launch vehicles. By combining trajectory simulation, atmospheric modeling, emission inventories, and optional integrating NASA's Chemical Equilibrium Analysis (CEA), it provides detailed insights into launch vehicle emissions. The program is highly customizable and supports both simplified and advanced emission calculations. 
LEAT is a scientific project and if you use it for publications or presentations in science, please support the project by citing the tool as following:
Jan-Steffen Fischer and Sebastian Winterhoff and Stefanos Fasoulas, Launch Emission Assessment Tool (LEAT), Version 1.1.0. University of Stuttgart. 2026. Available: https://github.com/lcasts/LEAT

---

## **Features**

- **Trajectory simulation**: Calculates detailed launch trajectories for each launch vehicle.
- **Atmospheric modeling**: Integrates environmental data from pymsis module for accurate analysis.
- **Emission analysis**: Computes emissions using Emission Factors, NASA CEA or Cantera, including afterburning with the entrained air.
- **Booster recovery**: Computes trajectory and emissions of returning boosters (RTLS, ADSL, SEA) from telemetry templates.
- **Customizable scenarios**: Easily specify launches, configurations, and analysis parameters.

---

## **How to Use**

### 1. **Include NASA CEA Software & Prepare Input Data**

- Include NASA CEA Software:

  - To calculate the emissions in more detail using NASA CEA you need to put all the software files of the Win Gui into the folder `/NASA_CEA`.
  - You can still calculate the trajectory and using emission factors, if you don't have access to the NASA CEA Software.
  - If you do not have access to a running version of NASA CEA you can request one over this link: `https://software.nasa.gov/software/LEW-17687-1`.

- Update all required data files in the `input_data` folder:

  - All Excel Files include an extra README sheet for further informations.
  - `config.py`: Core configuration file to customize analysis.
  - `emission_factors.xlsx`: Emission factors for simplified calculations.
  - `launch_sites.xlsx`: Definitions of launch sites.
  - `LV_Database.xlsx`: Database of launch vehicles (engines, fuel, force, etc.), default.
  - `scenarios.xlsx`: Definitions of launch scenarios (use cases), default.
  - `LV_Database_GMD.xlsx` and `scenarios_GMD_emission.xlsx`: Input of the GMD paper examples.
  - `trajectory_2023-013_Starlink5492_input.csv`: Example file if you want to skip the trajectory calculation by using already existing data.
  - `recovery/`: Booster return telemetry templates (only needed with `calculate_recovery = True`).

### 2. **Customize Configurations**

- Modify the `config.py` file to define your analysis:
  - **Scenarios**: Specify scenarios to process (`user_defined_scenarios` or all scenarios).
  - **Simulation**: Adjust numerical methods, tolerances, and simulation duration.
  - **Emission settings**: Choose between simplified emission factors or NASA CEA for calculations.
  - **Data compression**: Optimize runtime by compressing atmospheric data.
  - **Recovery**: Switch on the booster return with `calculate_recovery` (default `False`).
- Input files are set in `config.py` with `file_name_scenarios` and `file_name_launch_vehicle`, or without editing via the environment variables `LEAT_SCENARIO_FILE` and `LEAT_LV_DATABASE`.
- To rerun the GMD paper examples use `scenarios_GMD_emission.xlsx` and `LV_Database_GMD.xlsx`, e.g. in PowerShell:
  ```powershell
  $env:LEAT_SCENARIO_FILE = 'scenarios_GMD_emission.xlsx'; $env:LEAT_LV_DATABASE = 'LV_Database_GMD.xlsx'; python main.py
  ```

### 3. **Run the Program**

1. **Set Up the Virtual Environment** (Python >= 3.10)

   - Create a virtual environment (if not already created):
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

2. **Install Required Python Modules**

   - Install the dependencies listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Execute the Main Script**

   - Run the program:
     ```bash
     python main.py
     ```

4. **View the Results**
   - Results will be saved in the `output_data` folder:
     - **`data_raw/`**: Contains the raw data files.
     - **`emissions/`**: Contains emissions data for all timesteps in Excel format.
     - **`trajectory/`**: Contains the trajectory data for all timesteps in Excel format.

### 4. **Optional Utilities**

- To clean up the output folder:
  ```bash
  python cleanup_output_data.py
  ```

---

## **Folder Structure**

```
.
├── input_data/                 # Contains all input files
│   ├── cea_species.yaml
│   ├── config.py
│   ├── emission_factors.xlsx
│   ├── launch_sites.xlsx
│   ├── LV_Database.xlsx
│   ├── LV_Database_GMD.xlsx    # GMD paper examples
│   ├── scenarios.xlsx
│   ├── scenarios_GMD_emission.xlsx    # GMD paper examples
│   ├── trajectory_2023-013_Starlink5492_input.csv    # Optional
│   └── recovery/               # Booster return telemetry templates
├── NASA_CEA/                   # NASA CEA tool and related files
├── output_data/                # Folder for storing output results
│   ├── data_raw/               # Raw output files
│   ├── emissions/              # Emission results
│   ├── trajectory/             # Trajectory and atmospheric data
├── scripts/                    # Helper scripts
│   ├── data_processing.py
│   ├── emissions.py
│   ├── emissions_recovery.py   # Recovery emissions incl. bow shock
│   ├── trajectory.py
│   └── trajectory_recovery.py  # Recovery trajectory from templates
├── cleanup_output_data.py      # Script to clear output data
├── LICENSE.txt                 # Software License Agreement
├── main.py                     # Main entry point for the tool
├── README.md                   # Main README
└── requirements.txt            # Required Python Modules
```

---

## **Technical Notes**

1. **Input Data Requirements**:

   - Ensure the launch vehicle database contains aerodynamic variables incl. `A_e_A_R` (nozzle exit area / reference area); otherwise, the ODE solver will fail.
   - Optional column `UP` in the launch vehicle database: mass not covered by the stage data, carried like the payload (default 0).
   - Optional scenario columns (empty = nominal flight): `h_break_km` (ascent stops at this altitude), `t_recovery_end` (booster return stops at this flight time in s), `recovery_landing_burn` (0 = booster return ends before the landing burn).
   - Ensure `trajectory_input.csv` has the given column structure with t,h,date,lon,lat,v,d_m,active_phases,time_interval
   - For Black Carbon calculation, fuel names must match entries (BC) in `emission_factors.xlsx`.

2. **Simulation Details**:

   - The simulation assumes a multi-phase launch:
     - Phase 1: Both booster and main engine can be active when defined in LV_Database.
     - Phase 2: Only the main engine is active.
     - Phase 3-X: Further engines are active.
   - Boosters that burn longer than or as long as stage 1 are handled with own phase logic.
   - The trajectory stops at `height_ode_break` or when the last stage has burnt out.
   - Pressure, speed of sound, isentropic exponent and gas constant are taken from the local NRLMSIS gas mixture.

3. **NASA CEA Configuration**:
   - NASA CEA requires proper setup in the `NASA_CEA` folder.
   - Special fuels can be defined in more detail in `config.py` under `special_fuels`.

4. **Booster Recovery**:
   - The recovery mode is read from the launch vehicle name (`RTLS`, `ADSL` or `SEA`).
   - The template `input_data/recovery/<family>_<launch_site>_<type>_telemetry.csv` is selected, `<family>` being a prefix of the launch vehicle name. Site codes sharing one template are set in `recovery_site_aliases`.
   - The template is rotated onto the launch azimuth and moved onto the launch date; the descent afterburning runs against the air behind the bow shock (`recovery_shock`), unpowered supersonic rows add the air NOx of the shock.
   - Results are written as `<scenario>_<type>_recovery`. Falcon Heavy side boosters use the Falcon-9 template twice.

   | Template | Vehicle family | Launch site | Type | Reference flight date |
   |---|---|---|---|---|
   | `Falcon-9_CC-LC40_ADSL_telemetry.csv` | Falcon-9 | CC-LC40 | ADSL | 2025-02-18 |
   | `Falcon-9_CC-LC40_RTLS_telemetry.csv` | Falcon-9 | CC-LC40 | RTLS | 2025-11-02 |
   | `Falcon-9_KSC-LC39A_ADSL_telemetry.csv` | Falcon-9 | KSC-LC39A | ADSL | 2025-01-08 |
   | `Falcon-9_KSC-LC39A_RTLS_telemetry.csv` | Falcon-9 | KSC-LC39A | RTLS | 2025-08-22 |
   | `Falcon-9_VSFBS-SLC4E_ADSL_telemetry.csv` | Falcon-9 | VSFBS-SLC4E | ADSL | 2025-02-23 |
   | `Falcon-9_VSFBS-SLC4E_RTLS_telemetry.csv` | Falcon-9 | VSFBS-SLC4E | RTLS | 2025-09-22 |
   | `Falcon-9 v1.2 (Block5) Inmarsat_CC-LC40_ADSL_telemetry.csv` | Falcon-9 (Inmarsat 6 F2 use case) | CC-LC40 | ADSL | 2023-02-18 |
   | `Starship V1_STB-OLP1_SEA_telemetry.csv` | Starship (all splashdowns) | STB-OLP1 | SEA | 2024-06-06 |
   | `Starship V2_STB-OLP1_RTLS_telemetry.csv` | Starship V2 | STB-OLP1 | RTLS | 2025-03-06 |
   | `Starship_STB-OLP1_RTLS_telemetry.csv` | Starship | STB-OLP1 | RTLS | 2026-09-28 |

   Source of the telemetry: Extracted from public SpaceX webcast telemetry.

---

## **Use Cases**

The use cases are the scenarios of the default `scenarios.xlsx` (prefix `UC`), their vehicles are in `LV_Database.xlsx` (Falcon-9 entries with the dissertation values, e.g. `A_e_A_R = 0.5` and `UP`). To run a single use case set `all_scenarios = False` and `user_defined_scenarios` in `config.py`; variants are selected by config switches only. The dissertation use cases were calculated with `height_ode_break = 100e3` (default here: 120 km); set this value to reproduce them.

| Use case | Scenarios | Config switches |
|---|---|---|
| UC1 Ariane 6 CSO-3 | `UC1_2025-044_CSO-3_Ariane62` | `use_cantera = False`; variant per engine: `use_cantera = True`, `afterburning_temperature_per_engine = True`. Uses the public `Ariane 62` entry, so results differ from the dissertation. |
| UC2 Inmarsat 6 F2 (ADSL) | `UC2_2023-022_Inmarsat_6_F2_ADSL` | `calculate_recovery = True`; variant without bow shock: `recovery_shock = False` |
| UC3 Inmarsat 6 F2 recovery variants | `UC3_2023-022_Inmarsat_6_F2_RTLS`, `UC3_2023-022_Inmarsat_6_F2_Expended` | `use_cantera = False`, `calculate_recovery = True`; vary `x_ab` (e.g. 1.1 to 16) |
| UC4 Starlink 5492 recovery variants | `UC4_2023-013_Starlink5492_Expended`, `_ADSL`, `_RTLS` | `use_cantera = False`, `calculate_recovery = True`; external trajectory: `use_external_trajectory_data = True` with `trajectory_2023-013_Starlink5492_input.csv` |
| UC5 Starship flight 8 | `UC5_2025-F03_Starship_FT-8_RTLS` | `calculate_recovery = True` |

---

## **Acknowledgments**

- The trajectory calculation code is partly based on Nick Elvin Zeiger's Bachelor thesis:  
  _"Analysis of the environmental impact of rocket engine emissions as a function of trajectory"_  
  University of Stuttgart.

- The authors would like to gratefully acknowledge funding by the German Space Agency DLR, funding reference 50RL2180 from the Federal Ministry for Economic Affairs and Climate Action of Germany based on a decision of the German Bundestag.

---

## **Contact**
- If you have questions, remarks or wishes please contact us under fischerj[at]irs.uni-stuttgart.de

---

## Authors

- **Jan-Steffen Fischer** - [lcasts](https://github.com/lcasts)
- **Sebastian Winterhoff** - [SWinterhoff](https://github.com/SWinterhoff)
- **Stefanos Fasoulas**
