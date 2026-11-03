### Booster recovery (RTLS, ADSL, SEA): places a booster return template on the scenario and computes its emissions.

from input_data.config import *
import os
import pandas as pd
from ast import literal_eval
from scripts import trajectory as traj_module
from scripts import data_processing, emissions, emissions_recovery


def get_recovery_type(launch_vehicle: str) -> str | None:
    """Return 'RTLS', 'ADSL' or 'SEA' if it is part of the launch vehicle name, else None."""
    for rtype in ("RTLS", "ADSL", "SEA"):
        if rtype in str(launch_vehicle):
            return rtype
    return None


# Vehicles without own template: (template family, number of returning boosters, vehicle entry for the booster data).
TEMPLATE_FAMILY_OVERRIDE = {
    "Falcon Heavy Side RTLS": ("Falcon-9", 2, "Falcon-9 v1.2 (Block5) RTLS"),
    "Falcon Heavy Core ADSL": ("Falcon-9", 1, "Falcon-9 v1.2 (Block5) ADSL"),
    "Falcon Heavy": ("Falcon-9", 2, "Falcon-9 v1.2 (Block5) ADSL"),
}

# (vehicle prefix, recovery type) -> template family; all Starship splashdowns use the 'Starship V1' template.
TEMPLATE_BY_RECOVERY = {("Starship", "SEA"): "Starship V1"}


def recovery_booster_count(launch_vehicle: str) -> int:
    """Number of boosters returning on the same template (1 unless overridden)."""
    for prefix, (_family, n, _proxy) in TEMPLATE_FAMILY_OVERRIDE.items():
        if str(launch_vehicle).startswith(prefix):
            return n
    return 1


def recovery_vehicle(launch_vehicle: str) -> str:
    """Vehicle entry the return leg is computed with (the vehicle itself unless overridden)."""
    for prefix, (_family, _n, proxy) in TEMPLATE_FAMILY_OVERRIDE.items():
        if str(launch_vehicle).startswith(prefix):
            return proxy
    return launch_vehicle


def _find_template_path(launch_vehicle: str, launch_site: str, recovery_type: str) -> str:
    """Find input_data/recovery/{family}_{launch_site}_{recovery_type}_telemetry.csv, family being a prefix of the vehicle name."""
    launch_site = recovery_site_aliases.get(launch_site, launch_site)
    folder = os.path.join("input_data", "recovery")
    for prefix, (family, _n, _proxy) in TEMPLATE_FAMILY_OVERRIDE.items():
        if str(launch_vehicle).startswith(prefix):
            launch_vehicle = family
    for (prefix, rtype), family in TEMPLATE_BY_RECOVERY.items():
        if rtype == recovery_type and str(launch_vehicle).startswith(prefix):
            launch_vehicle = family
    suffix = f"_{launch_site}_{recovery_type}_telemetry.csv"

    candidates = [
        f for f in sorted(os.listdir(folder))
        if f.endswith(suffix) and str(launch_vehicle).startswith(f[: -len(suffix)])
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Recovery template not found for {launch_vehicle} / {launch_site} / {recovery_type}.\n"
            f"Expected a file '<family>{suffix}' in {os.path.abspath(folder)} "
            f"whose family is a prefix of the vehicle name."
        )
    return os.path.join(folder, candidates[0])


def _load_template(path: str) -> pd.DataFrame:
    """Load a recovery template and convert active_phases / time_interval back to lists."""
    df = pd.read_csv(path)
    df["active_phases"] = df["active_phases"].apply(
        lambda x: [] if x == "[]" else literal_eval(x) if isinstance(x, str) else x
    )
    df["time_interval"] = df["time_interval"].apply(
        lambda x: [] if x == "[]" else literal_eval(x) if isinstance(x, str) else x
    )
    return df


def _update_coordinates(
    df: pd.DataFrame, azimuth: float, lat_0: float, lon_0: float
) -> pd.DataFrame:
    """Recalculate lat/lon from the launch site, the launch azimuth and the template downrange."""
    df = df.copy()
    for idx, row in df.iterrows():
        downrange_m = row["downrange_km"] * 1000.0
        lat_new, _, lon_new, _ = traj_module.coordinates(azimuth, lat_0, lon_0, downrange_m)
        df.at[idx, "lat"] = lat_new
        df.at[idx, "lon"] = lon_new
    return df


def _update_datetime(df: pd.DataFrame, launch_date) -> pd.DataFrame:
    """Move the template onto the scenario launch date, so the atmosphere is that of the scenario."""
    df = df.copy()
    df["date"] = pd.to_datetime(launch_date) + pd.to_timedelta(df["t"], unit="s")
    return df


# Relative tolerance within which two rows burn at the same mass flow level
MASS_FLOW_LEVEL_RTOL = 1e-3
# A single row is a level of its own if a known level is an integer multiple of its mass flow
ENGINE_FRACTIONS = range(2, 11)
ENGINE_FRACTION_RTOL = 1e-4


def _resolve_burn_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give every burning row the mass flow the engines actually ran at.

    Altitude bins in which the engines ignite, shut down or change their number carry an averaged,
    too low mass flow, which would overstate the entrained air of the afterburning. Such rows are
    treated as a mixture of two mass flow levels: with 'engines off' the burn time is shortened,
    between two burning levels the row is split. The propellant of the template is conserved.
    """
    df = df.reset_index(drop=True)
    duration = df["time_interval"].apply(lambda iv: sum(float(b) - float(a) for a, b in iv))
    burning = (df["d_m"] > 0) & (duration > 0)
    flow = (df["d_m"] / duration).where(burning)

    def same(a, b):
        return abs(a - b) <= MASS_FLOW_LEVEL_RTOL * max(a, b)

    levels = []

    def is_new_level(value):
        return not any(same(value, level) for level in levels)

    for i in df.index[burning]:
        if i + 1 in df.index and burning[i + 1] and same(flow[i], flow[i + 1]) and is_new_level(flow[i]):
            levels.append(float(flow[i]))
    for i in df.index[burning]:
        fractions = [level / flow[i] for level in levels if level > flow[i]]
        if is_new_level(flow[i]) and any(
                round(f) in ENGINE_FRACTIONS and abs(f - round(f)) <= ENGINE_FRACTION_RTOL * f
                for f in fractions):
            levels.append(float(flow[i]))

    def level_of(i):
        """Level a row burns at; 0.0 without burn, None for a row between two levels."""
        if i not in df.index or not burning[i]:
            return 0.0
        return next((level for level in levels if same(flow[i], level)), None)

    def neighbour_level(i, step):
        """Level of the nearest row on a level in direction step, 0.0 beyond the burn."""
        i += step
        while level_of(i) is None:
            i += step
        return level_of(i)

    interpolated = [c for c in ("h", "v", "downrange_km") if c in df.columns]
    rows = []
    for i, row in df.iterrows():
        if level_of(i) is not None or len(row["time_interval"]) != 1:
            rows.append(row)
            continue

        first, second = neighbour_level(i, -1), neighbour_level(i, +1)
        if not min(first, second) < flow[i] < max(first, second):
            below = max([level for level in levels if level < flow[i]], default=0.0)
            above = [level for level in levels if level > flow[i]]
            if not above:
                rows.append(row)
                continue
            first, second = below, min(above)

        t_start, t_end = (float(x) for x in row["time_interval"][0])
        # Time spent at the first level, from first * t + second * (dt - t) = d_m
        t_first = float((flow[i] - second) / (first - second) * (t_end - t_start))
        t_change = t_start + t_first    # Plain floats, the intervals are written to the csv as text

        if first == 0.0:
            row = row.copy()
            row["time_interval"] = [[t_change, t_end]]
            rows.append(row)
        elif second == 0.0:
            row = row.copy()
            row["time_interval"] = [[t_start, t_change]]
            rows.append(row)
        else:
            part = row.copy()
            fraction = t_first / (t_end - t_start)
            if i > 0:
                for column in interpolated:
                    part[column] = df.at[i - 1, column] + fraction * (row[column] - df.at[i - 1, column])
                part["t"] = df.at[i - 1, "t"] + fraction * (row["t"] - df.at[i - 1, "t"])
            part["d_m"] = first * t_first
            part["time_interval"] = [[t_start, t_change]]
            rows.append(part)
            row = row.copy()
            row["d_m"] = row["d_m"] - part["d_m"]
            row["time_interval"] = [[t_change, t_end]]
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


RECOVERY_PHASE = "Phase_1"


def _force_booster_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Pin every active row to the booster phase, so the booster engine is used."""
    df = df.copy()
    df["active_phases"] = df["active_phases"].apply(
        lambda phases: [RECOVERY_PHASE] if phases else phases
    )
    return df


def _trim_to_return(df: pd.DataFrame) -> pd.DataFrame:
    """Cut the template to the return leg, starting at the first burn or the start of the descent."""
    burning = df.index[df["d_m"] > 0]
    descending = df.index[df["h"].diff() < 0]

    candidates = [idx.min() for idx in (burning, descending) if len(idx)]
    if not candidates:
        return df

    start = min(candidates)
    return df.loc[start:].reset_index(drop=True)


def _scenario_number(scenario, column):
    """Value of an optional scenario column, None if the column is missing or empty."""
    if column not in scenario.index:
        return None
    try:
        value = float(scenario[column])
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _cut_at_failure(df: pd.DataFrame, scenario) -> pd.DataFrame:
    """End the return leg early with the optional columns t_recovery_end and recovery_landing_burn."""
    landing_burn = _scenario_number(scenario, "recovery_landing_burn")
    if landing_burn is not None and landing_burn == 0:
        burning = df.index[df["d_m"] > 0]
        if len(burning):
            # First row of the last contiguous burn block
            start = burning[-1]
            while start - 1 in burning:
                start -= 1
            if start > burning[0]:      # Never cut a template with a single burn only
                df = df.loc[: start - 1]

    t_end = _scenario_number(scenario, "t_recovery_end")
    if t_end is not None:
        df = df[df["t"] <= t_end]

    return df.reset_index(drop=True)


def _scale_to_boosters(emission_results: pd.DataFrame, df_recovery: pd.DataFrame, n: int) -> pd.DataFrame:
    """Multiply d_m and all emission columns by the number of returning boosters."""
    if n == 1:
        return emission_results
    out = emission_results.copy()
    mass_cols = ["d_m"] + [c for c in out.columns if c not in df_recovery.columns and not c.startswith("T_ab")]
    for c in mass_cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c] * n
    return out


def compute_recovery(
    scenario_name: str,
    df_scenarios,
    df_launch_vehicles,
    df_launch_vehicles_engines,
    df_emission_factors,
) -> None:
    """Compute trajectory and emissions of the booster return of an RTLS, ADSL or SEA scenario."""
    scenario = df_scenarios.loc[scenario_name]
    recovery_type = get_recovery_type(scenario["launch_vehicle"])
    if recovery_type is None:
        return

    launch_vehicle_name = scenario["launch_vehicle"]
    launch_site_name = scenario["launch_site"]
    azimuth = scenario["azimuth"]
    lat_0 = scenario["lat"]
    lon_0 = scenario["lon"]
    recovery_scenario_name = f"{scenario_name}_{recovery_type}_recovery"
    n_boosters = recovery_booster_count(launch_vehicle_name)

    print(
        colors.BOLD + colors.BLUE
        + f"\n[Recovery] Processing {recovery_type} booster return for: {scenario_name}"
        + colors.END
    )

    # 1. Load template
    try:
        template_path = _find_template_path(launch_vehicle_name, launch_site_name, recovery_type)
        df_recovery = _load_template(template_path)
        print(colors.GREEN + f"[Recovery] Loaded template: {os.path.normpath(template_path)}" + colors.END)
        rows_template = len(df_recovery)
        df_recovery = _resolve_burn_transitions(df_recovery)
        if len(df_recovery) > rows_template:
            print(colors.GREEN + f"[Recovery] {len(df_recovery) - rows_template} rows split where "
                                 f"the mass flow level changes within an altitude bin" + colors.END)
    except FileNotFoundError as e:
        print(colors.RED + f"[Recovery] {e}" + colors.END)
        return

    # 2. Rotate onto the launch azimuth and move onto the scenario date
    df_recovery = _update_coordinates(df_recovery, azimuth, lat_0, lon_0)
    df_recovery = _update_datetime(df_recovery, scenario["date"])
    print(
        colors.GREEN
        + f"[Recovery] Coordinates updated (azimuth={azimuth}°, "
          f"lat_0={lat_0:.4f}, lon_0={lon_0:.4f}), "
          f"date set to {df_recovery['date'].iloc[0]}"
        + colors.END
    )

    df_recovery = _force_booster_phase(df_recovery)

    rows_before = len(df_recovery)
    df_recovery = _trim_to_return(df_recovery)
    rows_return = len(df_recovery)
    df_recovery = _cut_at_failure(df_recovery, scenario)
    if df_recovery.empty:
        print(colors.YELLOW + "[Recovery] Return leg is empty after the cut at the failure, skipped." + colors.END)
        return
    if len(df_recovery) < rows_return:
        print(
            colors.YELLOW
            + f"[Recovery] Booster lost on the way back: return leg ends at t={df_recovery['t'].iloc[-1]:.1f} s "
              f"(h={df_recovery['h'].iloc[-1]:.1f} km), {rows_return - len(df_recovery)} rows dropped"
            + colors.END
        )
    print(
        colors.GREEN
        + f"[Recovery] Return leg from t={df_recovery['t'].iloc[0]:.1f} s "
          f"(h={df_recovery['h'].iloc[0]:.1f} km), "
          f"{len(df_recovery)} of {rows_before} rows, "
          f"propellant {df_recovery['d_m'].sum():.0f} kg"
          + (f" per booster, {n_boosters} boosters (emissions scaled x{n_boosters})" if n_boosters > 1 else "")
        + colors.END
    )

    # 3. Save the recovery trajectory
    try:
        recovery_traj_path = os.path.join(
            output_data_folder_path_trajectory,
            f"{output_data_trajectory_name}{output_data_raw_name}{recovery_scenario_name}.csv",
        )
        df_recovery.to_csv(recovery_traj_path, index=False)
        print(colors.GREEN + f"[Recovery] Trajectory saved: {os.path.normpath(recovery_traj_path)}" + colors.END)
    except Exception as e:
        print(colors.RED + f"[Recovery] Could not save trajectory: {e}" + colors.END)

    # The emission functions look the scenario up by name, so the return gets its own entry.
    if recovery_scenario_name not in df_scenarios.index:
        df_scenarios = df_scenarios.copy()
        df_scenarios.loc[recovery_scenario_name] = df_scenarios.loc[scenario_name]
    # Use the single-booster entry for vehicles like Falcon Heavy.
    proxy = recovery_vehicle(launch_vehicle_name)
    if proxy != launch_vehicle_name:
        df_scenarios.loc[recovery_scenario_name, "launch_vehicle"] = proxy
        print(colors.GREEN + f"[Recovery] Booster data from '{proxy}'" + colors.END)

    # 4. Build phases (same call as main.py)
    try:
        df_phases = data_processing.phases(recovery_scenario_name, df_scenarios, df_launch_vehicles)
    except Exception as e:
        print(colors.RED + f"[Recovery] phases() failed: {e}" + colors.END)
        return

    # 5. Emission calculations
    if not calculate_emissions:
        return

    try:
        if use_emission_factors:
            emission_results = emissions.emission_factors(
                df_recovery, df_phases, df_scenarios, df_launch_vehicles,
                df_emission_factors, recovery_scenario_name,
            )
            emission_results = _scale_to_boosters(emission_results, df_recovery, n_boosters)
            data_processing.save_emission_results(emission_results, recovery_scenario_name, "Emission_Factors")

        # In the descent the afterburning runs against the air behind the bow shock.
        if use_nasa_cea:
            if rocket_equilibrium:
                emission_results = emissions_recovery.run_nasa_cea_recovery(
                    df_recovery, df_phases, df_scenarios, df_launch_vehicles_engines,
                    df_launch_vehicles, recovery_scenario_name, df_emission_factors, "equilibrium",
                )
                emission_results = _scale_to_boosters(emission_results, df_recovery, n_boosters)
                data_processing.save_emission_results(emission_results, recovery_scenario_name, "NASA_CEA_Equilibrium")
            if rocket_frozen:
                emission_results = emissions_recovery.run_nasa_cea_recovery(
                    df_recovery, df_phases, df_scenarios, df_launch_vehicles_engines,
                    df_launch_vehicles, recovery_scenario_name, df_emission_factors, "frozen",
                )
                emission_results = _scale_to_boosters(emission_results, df_recovery, n_boosters)
                data_processing.save_emission_results(emission_results, recovery_scenario_name, "NASA_CEA_Frozen")

        if use_cantera:
            emission_results = emissions_recovery.run_cantera_recovery(
                df_recovery, df_phases, df_scenarios, df_launch_vehicles_engines,
                df_launch_vehicles, recovery_scenario_name, df_emission_factors,
            )
            emission_results = _scale_to_boosters(emission_results, df_recovery, n_boosters)
            data_processing.save_emission_results(emission_results, recovery_scenario_name, "Cantera")

    except Exception as e:
        print(colors.RED + f"[Recovery] Emission calculation error: {e}" + colors.END)
