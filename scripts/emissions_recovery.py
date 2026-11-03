### Recovery emissions: the afterburning of the descending booster runs against the air behind the bow shock.

import math
import re
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pymsis import msis

from input_data.config import *
from scripts import data_processing
from scripts import emissions

# Shock settings are defined in input_data/config.py (Recovery Config).

#region: 1. Atmosphere of the local mixture
def calculate_atmosphere_shock(row):
    """Atmosphere state incl. speed of sound, kappa and R of the local NRLMSIS mixture."""
    atmosphere = msis.run(row["date"], row["lon"], row["lat"], row["h"])

    return data_processing.atmosphere_mixture(atmosphere)
#endregion

#region: 2. Normal shock
def compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix):
    """Perfect-gas normal shock with kappa and R of the local mixture (p_2 has the unit of p_atm)."""
    kappa = kappa_mix
    mach_1 = velocity / c_atm

    isentropic_factor_1 = 1.0 + 0.5 * (kappa - 1.0) * mach_1**2
    t_total_1 = t_atm * isentropic_factor_1
    p_total_1 = p_atm * isentropic_factor_1 ** (kappa / (kappa - 1.0))

    if mach_1 <= 1.0:
        return (mach_1, t_total_1, p_total_1, mach_1,
                p_atm, rho_atm, t_atm, t_total_1, p_total_1, velocity)

    mach_2 = math.sqrt(
        (1.0 + 0.5 * (kappa - 1.0) * mach_1**2)
        / (kappa * mach_1**2 - 0.5 * (kappa - 1.0))
    )
    pressure_ratio = 1.0 + (2.0 * kappa / (kappa + 1.0)) * (mach_1**2 - 1.0)
    density_ratio = ((kappa + 1.0) * mach_1**2) / ((kappa - 1.0) * mach_1**2 + 2.0)

    p_2 = p_atm * pressure_ratio
    rho_2 = rho_atm * density_ratio
    t_2 = t_atm * (pressure_ratio / density_ratio)

    t_total_2 = t_total_1
    p_total_2 = p_2 * (1.0 + 0.5 * (kappa - 1.0) * mach_2**2) ** (kappa / (kappa - 1.0))
    velocity_2 = mach_2 * math.sqrt(kappa * r_mix * t_2)

    return (mach_1, t_total_1, p_total_1, mach_2,
            p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2)
#endregion

#region: 3. NASA CEA calculation of the shocked air
# Only the prefix is matched, because CEA2 misspells the frozen header ("CONDITI1ONS").
SHOCK_BASIS_HEADER = {
    "eq": "EQUILIBRIUM COMPOSITION FOR INCIDENT SHOCKED CONDITI",
    "fz": "FROZEN COMPOSITION FOR INCIDENT SHOCKED CONDITI",
}
SHOCK_STATIONS = {
    "inc_eq": ("eq", "SHOCKED GAS (2)--INCIDENT--EQUILIBRIUM"),
    "inc_fz": ("fz", "SHOCKED GAS (2)--INCIDENT--FROZEN"),
    "ref_eq": ("eq", "SHOCKED GAS (5)--REFLECTED--EQUILIBRIUM"),
    "ref_fz": ("fz", "SHOCKED GAS (5)--REFLECTED--FROZEN"),
}


def cea_float(token):
    """Read the CEA short notation ('1.8832-4' -> 1.8832e-4)."""
    text = str(token).strip().replace(" ", "")
    if not text:
        return float("nan")
    if "*" in text:
        # An overflowed mantissa with exponent ('*****-10') stands for 10^exp.
        match = re.match(r"^\*+([+-]\d+)$", text)
        if match:
            return 10.0 ** int(match.group(1))
        return 1.0             # Overflow without exponent is of order 1
    try:
        return float(text)
    except ValueError:
        pass
    match = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", text)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    return float("nan")


def create_shock_inp(filename, t_atm, p_atm_bar, df_air_massfractions,
                     velocity, mach_1, problem_override=None):
    """Write the CEA input file for the shocked air."""
    problem = problem_override or problem_hotgas_calculation

    if problem == "shock":
        if not np.isfinite(mach_1) or mach_1 < shock_mach_min:
            raise ValueError(
                f"Mach 1 = {mach_1} is below shock_mach_min = {shock_mach_min}; "
                "NASA CEA has no shock solution here."
            )
        if shock_input == "u1" and (velocity is None or not np.isfinite(velocity)):
            raise ValueError("shock_input = 'u1' requires the free-stream velocity")

    air_species = {
        s: df_air_massfractions[s].values[0] for s in df_air_massfractions.columns
    }
    # Reactants without mass make the CEA shock problem fail ('M6202: MATH - log: SING error').
    air_species = {k: v for k, v in air_species.items() if float(v) > 0.0}
    if len(air_species) > 24:
        air_species = {k: v for k, v in air_species.items() if k not in ["He", "Ar"]}

    if problem == "shock":
        # The inflow state is the undisturbed atmosphere.
        line = "problem  case=shock{} \n".format(" ions" if shock_ions else "")
        if shock_input == "u1":
            free_stream = f"u1={velocity}"
        elif shock_input == "mach1":
            free_stream = f"mach1={mach_1}"
        else:
            raise ValueError(f"Unknown shock_input: {shock_input!r}")
        line += "  shock  {}{}  {}, t,k={}, p,bar={}, \n".format(
            shock_flags, "  deb" if shock_debug else "", free_stream, t_atm, p_atm_bar
        )
        react_temperature = t_atm
    else:
        line = "problem,\n"
        line += f"  {problem}  p,bar={p_atm_bar} \n"
        react_temperature = t_atm

    line += "react \n"
    for species, mass_fraction in air_species.items():
        line += f"  name={species} wt={mass_fraction} t,k={react_temperature} \n"

    line += "output \n"
    line += f"  siunits massf trace={trace} \n"
    line += "  plot p t mach H rho s \n"
    line += "end \n"

    with open(filename + ".inp", "w") as file:
        file.write(line)


def read_shock_out(filename, station=None):
    """Mass fractions of one station of the shock problem as a Series."""
    station = station or shock_state
    basis, heading = SHOCK_STATIONS[station]
    block_header = SHOCK_BASIS_HEADER[basis]

    with open(filename + ".out") as file:
        text = file.read()

    start = text.find(block_header)
    if start < 0:
        raise ValueError(f"Block '{block_header}' not found in {filename}.out")

    section = text[start:]
    mass_start = section.find("MASS FRACTIONS")
    if mass_start < 0:
        raise ValueError(f"No MASS FRACTIONS in {filename}.out")

    fractions = {}
    for line in section[mass_start:].splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("*") and "THERMODYNAMIC" in stripped:
            break
        parts = stripped.split()
        if len(parts) < 2:
            continue
        name = emissions_clean_name(parts[0])
        value = cea_float(parts[1])
        if np.isfinite(value) and value > 0:
            fractions[name] = value
        if len(fractions) > 0 and stripped.startswith("PRODUCTS WHICH"):
            break

    if not fractions:
        raise ValueError(f"No species in station '{station}' of {filename}.out")

    return pd.Series(fractions)


def emissions_clean_name(name):
    return str(name).strip().lstrip("*")


def run_hotgas_cea(filename, t_atm, p_atm_bar, df_air_massfractions, velocity, mach_1):
    """Equilibrium of the shocked air, with 'hp' as fallback for 'shock'."""
    problems = [problem_hotgas_calculation]
    if problem_hotgas_calculation == "shock":
        problems.append("hp")

    last_error = None
    for problem in problems:
        try:
            create_shock_inp(filename, t_atm, p_atm_bar, df_air_massfractions,
                             velocity, mach_1, problem_override=problem)
            process = subprocess.Popen(
                [nasa_cea_folder_path + nasa_cea_exe],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            process.stdin.write(f"{filename}\n".encode())
            process.stdin.close()
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(process.stderr.read().decode().strip())

            if problem == "shock":
                species = read_shock_out(filename)
            else:
                output = emissions.readCEA(problem, filename)
                # Drop the state rows that readCEA puts in front of the species.
                state_rows = ["p", "t", "mach", "H", "rho", "s"]
                output = output.drop(index=[r for r in state_rows if r in output.index])
                species = pd.to_numeric(output["exit"], errors="coerce").dropna()
                species.index = [emissions_clean_name(i) for i in species.index]
                species = species[species > 0]
            return species, problem
        except Exception as error:
            last_error = error
            if problem != problems[-1]:
                print(f"NASA CEA '{problem}' failed for {filename}: {error} "
                      "- retrying with 'hp'.")
        finally:
            emissions.clean_directory(filename)

    raise last_error


def hotgas_state(row, tag="HOTGAS"):
    """Post-shock state and composition for one trajectory row, None in subsonic flow."""
    rho_atm, t_atm, p_atm, c_atm, df_air, kappa_mix, r_mix = (
        calculate_atmosphere_shock(row)
    )
    velocity = float(row["v"])

    (mach_1, _t_total_1, _p_total_1, _mach_2,
     p_2, rho_2, t_2, t_total_2, p_total_2, _velocity_2) = compute_shockwave(
        t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix
    )

    if mach_1 <= 1.0:
        return None

    species, problem = run_hotgas_cea(
        tag, t_atm, p_atm * 1e-5, df_air, velocity, mach_1
    )

    if afterburning_state == "total":
        t_used, p_used = t_total_2, p_total_2
    else:
        t_used, p_used = t_2, p_2

    return {
        "rho": rho_2,
        "rho_atm": rho_atm,
        "t": t_used,
        "p": p_used,
        "species": species,
        "mach_1": mach_1,
        "t_atm": t_atm,
        "p_atm": p_atm,
        "problem": problem,
    }
#endregion

#region: 4. Wrappers around the launch emission calculation
def descending_rows(df_trajectory):
    """Indices of the rows in which the stage loses altitude."""
    if df_trajectory is None:
        return None
    return set(df_trajectory.index[df_trajectory["h"].diff() < 0])


def row_timesteps(df_trajectory):
    """Time step [s] per row; the first row takes the second step."""
    dt = df_trajectory["t"].diff()
    return dt.where(dt > 0).bfill().fillna(0.0)


def reference_area(df_launch_vehicles, launch_vehicle):
    """Stage cross section [m^2] from the diameter in the LV database."""
    diameter = float(df_launch_vehicles.loc[launch_vehicle, "d"])
    return np.pi * diameter**2 / 4.0


def smart_round(number):
    """Round to the 5 significant digits of the CEA output to avoid spurious differences."""
    number = float(number)
    if not np.isfinite(number) or number == 0.0:
        return number
    return float(f"{number:.4e}")


def air_species_ei(shocked_species, df_air_massfractions):
    """Increase of the mass fractions from undisturbed to shocked air [kg/kg], mainly NO."""
    x_shocked = shocked_species.copy()
    x_shocked.index = [str(i).lstrip("*") for i in x_shocked.index]

    x_atm = df_air_massfractions.iloc[0].astype(float)
    x_atm.index = [str(i).lstrip("*") for i in x_atm.index]

    combined = pd.concat(
        [x_shocked.rename("shocked"), x_atm.rename("atm")], axis=1
    ).fillna(0.0)
    delta = combined["shocked"].map(smart_round) - combined["atm"].map(smart_round)
    return delta[delta > threshold]


def air_nox_emissions(df_trajectory, area, tag="AIRNOX"):
    """Air NOx [kg] of unpowered supersonic descent rows, with the air mass rho*v*A_ref*(x_ab-1)*dt."""
    descending = descending_rows(df_trajectory)
    timesteps = row_timesteps(df_trajectory)

    records = []
    for idx, row in df_trajectory.iterrows():
        if idx not in descending or float(row["d_m"]) > 0:
            continue

        rho_atm, t_atm, p_atm, c_atm, df_air, kappa_mix, r_mix = (
            calculate_atmosphere_shock(row)
        )
        velocity = float(row["v"])
        mach_1 = velocity / c_atm
        if mach_1 <= 1.0:
            continue

        (_m1, _tt1, _pt1, _m2, p_2, _rho_2, t_2,
         _tt2, _pt2, _v2) = compute_shockwave(
            t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix
        )
        if min(23999.0, t_2) <= recovery_nox_t_min:
            continue

        try:
            species, _problem = run_hotgas_cea(
                f"{tag}_{idx}", t_atm, p_atm * 1e-5, df_air, velocity, mach_1
            )
        except Exception as error:
            print(f"Air NOx skipped (t={row['t']:.1f} s, "
                  f"h={row['h']:.1f} km): {error}")
            continue

        ei = air_species_ei(species, df_air)
        if ei.empty:
            continue

        air_mass = rho_atm * velocity * area * (x_ab - 1.0) * float(timesteps[idx])
        record = (ei * air_mass).to_dict()
        record["t"] = row["t"]
        record["h"] = row["h"]
        record["row"] = idx
        records.append(record)

    return pd.DataFrame(records)


@contextmanager
def shocked_atmosphere(df_trajectory=None):
    """Temporarily replace the afterburning air by the shocked air for supersonic descent rows."""
    original = data_processing.calculate_atmosphere_data
    shock_rows = descending_rows(df_trajectory)

    def patched(row):
        if shock_rows is not None and row.name not in shock_rows:
            return original(row)

        try:
            state = hotgas_state(row)
        except Exception as error:
            print(f"Shock calculation failed (h={row['h']:.1f} km, "
                  f"v={row['v']:.0f} m/s): {error} - using the undisturbed atmosphere.")
            return original(row)

        if state is None:
            return original(row)

        df_species = pd.DataFrame([state["species"].to_dict()])
        # Emission indices refer to the undisturbed air, so shock-formed NO counts as emission.
        df_species.attrs["reference_air"] = original(row)[3]
        # Free-stream density and ambient pressure with the post-shock temperature and species.
        return state["rho_atm"], state["t"], state["p_atm"], df_species

    data_processing.calculate_atmosphere_data = patched
    try:
        yield
    finally:
        data_processing.calculate_atmosphere_data = original


def add_air_nox(results, df_air_nox):
    """Add the air NOx to the engine emissions, row by row of the trajectory."""
    if df_air_nox.empty:
        return results

    results = results.copy()
    air_nox = df_air_nox.set_index("row").drop(columns=["t", "h"]).fillna(0.0)
    for species in air_nox.columns:
        if species not in results.columns:
            results[species] = 0.0
        results[species] = results[species].fillna(0.0)
        results.loc[air_nox.index, species] += air_nox[species]
    return results


def run_nasa_cea_recovery(df_trajectory_results_cleaned, df_phases, df_scenarios,
                          df_launch_vehicles_engines, df_launch_vehicles,
                          scenario_name, df_emission_factors, rocket_case):
    """emissions.run_nasa_cea with shocked-air afterburning and air NOx if recovery_shock is set."""
    if not recovery_shock:
        return emissions.run_nasa_cea(
            df_trajectory_results_cleaned, df_phases, df_scenarios,
            df_launch_vehicles_engines, df_launch_vehicles,
            scenario_name, df_emission_factors, rocket_case,
        )

    with shocked_atmosphere(df_trajectory_results_cleaned):
        results = emissions.run_nasa_cea(
            df_trajectory_results_cleaned, df_phases, df_scenarios,
            df_launch_vehicles_engines, df_launch_vehicles,
            scenario_name, df_emission_factors, rocket_case,
        )

    launch_vehicle = df_scenarios.loc[scenario_name, "launch_vehicle"]
    area = reference_area(df_launch_vehicles, launch_vehicle)

    # run_nasa_cea removes thermo.lib and trans.lib, the following CEA runs need them again.
    emissions.init_directory()
    try:
        df_air_nox = air_nox_emissions(df_trajectory_results_cleaned, area)
    finally:
        emissions.clean_up()

    if not df_air_nox.empty:
        species_sum = df_air_nox.drop(columns=["t", "h", "row"]).sum()
        print(f"[Recovery] Air NOx from {len(df_air_nox)} unpowered rows "
              f"(A_ref={area:.2f} m2, x_ab={x_ab}): "
              f"{species_sum.sum():.3f} kg, thereof NO {species_sum.get('NO', 0.0):.3f} kg")

    return add_air_nox(results, df_air_nox)


def run_cantera_recovery(df_trajectory_results_cleaned, df_phases, df_scenarios,
                         df_launch_vehicles_engines, df_launch_vehicles,
                         scenario_name, df_emission_factors):
    """emissions.run_cantera with shocked-air afterburning and air NOx if recovery_shock is set."""
    if not recovery_shock:
        return emissions.run_cantera(
            df_trajectory_results_cleaned, df_phases, df_scenarios,
            df_launch_vehicles_engines, df_launch_vehicles,
            scenario_name, df_emission_factors,
        )

    launch_vehicle = df_scenarios.loc[scenario_name, "launch_vehicle"]
    area = reference_area(df_launch_vehicles, launch_vehicle)

    # The shock calculation needs thermo.lib and trans.lib, which run_cantera does not create.
    emissions.init_directory()
    try:
        with shocked_atmosphere(df_trajectory_results_cleaned):
            results = emissions.run_cantera(
                df_trajectory_results_cleaned, df_phases, df_scenarios,
                df_launch_vehicles_engines, df_launch_vehicles,
                scenario_name, df_emission_factors,
            )
        # Recreate the libraries in case the emission calculation removed them.
        emissions.clean_up()
        emissions.init_directory()
        df_air_nox = air_nox_emissions(df_trajectory_results_cleaned, area)
    finally:
        emissions.clean_up()

    if not df_air_nox.empty:
        species_sum = df_air_nox.drop(columns=["t", "h", "row"]).sum()
        print(f"[Recovery] Cantera: air NOx from {len(df_air_nox)} unpowered rows "
              f"(A_ref={area:.2f} m2, x_ab={x_ab}): "
              f"{species_sum.sum():.3f} kg, thereof NO {species_sum.get('NO', 0.0):.3f} kg")

    if results is None:
        return results
    return add_air_nox(results, df_air_nox)
#endregion
