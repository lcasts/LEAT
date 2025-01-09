# **LEAT**

### _Launch Emission Assessment Tool_

LEAT is a Python-based tool licensed under the GNU Affero General Public License v3.0. The licence can be found in LICENCE.md. It is designed to calculate the global emissions of specified launch vehicles. By combining trajectory simulation, atmospheric modeling, emission inventories, and optional integrating NASA's Chemical Equilibrium Analysis (CEA), it provides detailed insights into launch vehicle emissions. The program is highly customizable and supports both simplified and advanced emission calculations. 
LEAT is a scientific project and if you use it for publications or presentations in science, please support the project by citing the tool as following:
Jan-Steffen Fischer and Sebastian Winterhoff and Stefanos Fasoulas, Launch Emission Assessment Tool (LEAT). University of Stuttgart. 2025. Available: https://github.com/lcasts/LEAT

---

## **Features**

- **Trajectory simulation**: Calculates detailed launch trajectories for each launch vehicle.
- **Atmospheric modeling**: Integrates environmental data from pymsis module for accurate analysis.
- **Emission analysis**: Computes emissions either using Emission Factors or NASA CEA.
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
  - `LV_Database.xlsx`: Database of launch vehicles (engines, fuel, force, etc.).
  - `scenarios.xlsx`: Definitions of launch scenarios.
  - `trajectory_results_cleaned.csv`: Example file if you want to skip the trajectory calculation by using already existing data.

### 2. **Customize Configurations**

- Modify the `config.py` file to define your analysis:
  - **Scenarios**: Specify scenarios to process (`user_defined_scenarios` or all scenarios).
  - **Simulation**: Adjust numerical methods, tolerances, and simulation duration.
  - **Emission settings**: Choose between simplified emission factors or NASA CEA for calculations.
  - **Data compression**: Optimize runtime by compressing atmospheric data.

### 3. **Run the Program**

1. **Set Up the Virtual Environment**

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
│   ├── config.py
│   ├── emission_factors.xlsx
│   ├── launch_sites.xlsx
│   ├── LV_Database.xlsx
│   ├── scenarios.xlsx
│   └── trajectory_results_cleaned.csv    # Optional
├── NASA_CEA/                   # NASA CEA tool and related files
├── output_data/                # Folder for storing output results
│   ├── data_raw/               # Raw output files
│   ├── emissions/              # Emission results
│   ├── trajectory/             # Trajectory and atmospheric data
├── scripts/                    # Helper scripts
│   ├── data_processing.py
│   ├── emissions.py
│   └── trajectory.py
├── cleanup_output_data.py      # Script to clear output data
├── LICENSE.txt                 # Software License Agreement
├── main.py                     # Main entry point for the tool
├── README.md                   # Main README
└── requirements.txt            # Required Python Modules
```

---

## **Technical Notes**

1. **Input Data Requirements**:

   - Ensure `LV_Database.xlsx` contains aerodynamic variables; otherwise, the ODE solver will fail.
   - For Black Carbon calculation, fuel names must match entries (BC) in `emission_factors.xlsx`.

2. **Simulation Details**:

   - The simulation assumes a multi-phase launch:
     - Phase 1: Both booster and main engine can be active when defined in LV_Database.
     - Phase 2: Only the main engine is active.
     - Phase 3-X: Further engines are active.

3. **NASA CEA Configuration**:
   - NASA CEA requires proper setup in the `NASA_CEA` folder.
   - Special fuels can be defined in more detail in `config.py` under `special_fuels`.

---

## **Acknowledgments**

- The trajectory calculation code is partly based on Nick Elvin Zeiger's Bachelor thesis:  
  _"Analysis of the environmental impact of rocket engine emissions as a function of trajectory"_  
  University of Stuttgart.

- The authors would like to gratefully acknowledge funding by the German Space Agency DLR, funding reference 50RL2180 from the Federal Ministry for Economic Affairs and Climate Action of Germany based on a decision of the German Bundestag.

---

## **Acknowledgments**
- If you have questions, remarks or wishes please contact us under fischerj[at]irs.uni-stuttgart.de

---

## Authors

- **Jan-Steffen Fischer** - [lcasts](https://github.com/lcasts)
- **Sebastian Winterhoff** - [SWinterhoff](https://github.com/SWinterhoff)
- **Stefanos Fasoulas**
