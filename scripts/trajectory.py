import math
import numpy as np
from pymsis import msis
from input_data.config import *
from scipy.integrate import solve_ivp
from scripts.data_processing import handle_error

#region: Functions necessary for solving the trajectory ODE
def drag_coefficient(temp, v, rho, launch_vehicle):
    kappa = 1.4
    R = 287.1
    c = np.sqrt(kappa*R*temp)
    ma = v/c
    l = launch_vehicle['l']
    d = launch_vehicle['d']
    l_n = launch_vehicle['l_n']
    d_n = launch_vehicle['d_n']
    d_f = launch_vehicle['d_f']
    a_r = launch_vehicle['A_R']
    
    # Gleichung nach Fleeman, siehe Dokumentation
    if ma<1 and v!= 0:
        c_w_bf = 0.053*(l/d)*(ma/(0.5*rho*0.0624*(abs(v)*3.2808)**2 * (l*3.2808)))**0.2
        c_w_bp = (1-0.556435354)*(0.12+0.13*abs(ma)**2)
        c_w_bw = 0
        c_w = c_w_bf + c_w_bp + c_w_bw
        #c_w = 0.053*(l/d)*(ma*l/(0.5*rho*v**2))**0.2
    elif v == 0:
        c_w = 0
    elif l_n == 0 and ma<1:
        c_w = 0.15
    elif l_n == 0 and ma>1:
        c_w = 0.4
    else:
        c_w_bf = 0.053*(l/d)*(ma/(0.5*rho*0.0624*(v*3.2808)**2 * (l*3.2808)))**0.2
        c_w_bp = (1-0.556435354)*(0.25/ma)
        c_w_bw = (1.59 + 1.83/(ma**2))*(np.arctan(0.5/(l_n/d_n)))**1.69 * (1-(d_n/d_f)**2) + ((1.59 + 1.83/(ma**2))*(np.arctan(0.5/(0.5)))**1.69) * (d_n/d_f)**2
        #c_w_bw = (1.586 + 1.834/(ma**2))*(np.arctan(0.5/(l_n/d_n)))**1.69 * ((d_f**2-d_n**2)/(d_f**2)) + 0.665 * (1.59 + 1.83/(ma**2)) * (d_n**2)/(d_f**2)
        c_w = c_w_bf + c_w_bp + c_w_bw
    return c_w

def drag_force(c_w, rho, v, launch_vehicle):
    A_R = launch_vehicle['A_R']
    F_W = c_w*0.5*rho*v**2*A_R
    
    return F_W

def coordinates(inc, lat_1, lon_1, dx):
    inc = inc*np.pi/180
    c = ((dx/(R_0*2*np.pi))*360)*np.pi/180

    #spärische Trigonometrie, Regel von Neper
    #a = delta lon b = delta lat

    #tan(a) = tan(c)*cos(beta)
    a = math.atan((math.tan(c)*math.cos(inc)))
    a = a*180/np.pi
    lon_2 = (lon_1+a)

    #cos(b*) = cos(90-b) = sin(b) = sin(c) * sin(beta)
    b = math.asin((math.sin(c)*math.sin(inc)))
    b = b*180/np.pi
    lat_2 = (lat_1+b)

    # Differential does not need to be specified since the data is manual calculated
    dlat = 0
    dlon = 0

    return(lat_2, dlat, lon_2, dlon)

def gravity(g_0,R_0,h):
    g = g_0 * (R_0)**2/(R_0 + h)**2
    
    return g

def get_current_phase(t, phase_intervals):
    # Allow for small negative values by using a tolerance
    tolerance = max_step
    if t < phase_intervals[0] - tolerance:
        raise ValueError(f"Time t ({t}) is smaller than the start of the first phase interval ({phase_intervals[0]}).")
    elif t <= phase_intervals[0] + tolerance:
        return 'Phase_1'
    elif t >= phase_intervals[-1]:
        return 'Phase_X'    # No Engine active anymore
    
    idx = np.searchsorted(phase_intervals, t) - 1
    return f'Phase_{idx + 1}'

def get_phases(t, max_step, phases, phase_intervals):
    # Current Phase: 
    current_phase_name = get_current_phase(t, phase_intervals)
    if current_phase_name == "Phase_X":
        current_phase = None
    else:
        current_phase = phases.loc[current_phase_name]
    
    # Previous Phase:
    if current_phase_name != 'Phase_1' and current_phase_name != "Phase_X":
        prev_phase_name = f'Phase_{int(current_phase_name.split("_")[1]) - 1}'
        prev_phase = phases.loc[prev_phase_name]
    elif current_phase_name == "Phase_X":
        prev_phase_name = f'Phase_{len(phase_intervals) - 1}'
        prev_phase = phases.loc[prev_phase_name]
    else:
        prev_phase_name = None
        prev_phase = None
        
    # Last Timestep Phase:
    if t == 0:
        last_timestep_phase_name = current_phase_name
        last_timestep_phase = current_phase
    else:
        last_timestep_phase_name = get_current_phase(t - max_step, phase_intervals)
        if last_timestep_phase_name == "Phase_X":
            last_timestep_phase = None
        else:
            last_timestep_phase = phases.loc[last_timestep_phase_name]

    # Phase Changed Variable:
    if prev_phase_name == last_timestep_phase_name:
        phase_changed = True
    else:
        phase_changed = False
    
    return phase_changed, current_phase, prev_phase

def calculate_force(current_phase, pressure_ratio, h, scenario):
    if current_phase is None:
        F = 0
        force_reduced = False 
        return F, force_reduced
    
    # Init Force Data
    booster_force_sl = current_phase["booster_force_sl"]
    booster_force_vac = current_phase["booster_force_vac"]
    stage_force_sl = current_phase["stage_force_sl"]
    stage_force_vac = current_phase["stage_force_vac"]
    
    # Sea Level & Vacuum Forces depending on current phase (With ot without booster)
    # Only Stage Force available
    if np.isnan(booster_force_sl) or np.isnan(booster_force_vac):
        force_sl = stage_force_sl
        force_vac = stage_force_vac
    # Only Booster Force available
    elif np.isnan(stage_force_sl) or np.isnan(stage_force_vac):
        force_sl = booster_force_sl
        force_vac = booster_force_vac
    # Both Forces available
    else:
        force_sl = booster_force_sl + stage_force_sl
        force_vac = booster_force_vac + stage_force_vac
        
    # Total force equation: F(h(p)) = F_sl + (F_vac - F_sl) * (1 - p_a/p_0)
    F = force_sl + (force_vac - force_sl) * (1 - pressure_ratio)
    force_reduced = False
    
    # Check if altitude is within the specified range for force reduction
    if scenario["h_fr_start"] <= h <= scenario["h_fr_end"]:
        # Apply force reduction for Max-Q
        F *= scenario["force_reduced"]
        force_reduced = True
    
    # Differential not necessary when simply overwriting data
    dF = 0
    
    return F, F, dF

def calculate_massflow(current_phase):
    if current_phase is None:
        massflow = 0
        return massflow
    
    # Time Intervall of current phase
    time_interval = current_phase["time_end"] - current_phase["time_start"]
    
    # Total fuel mass
    booster_fuel_mass = current_phase["booster_fuel_mass"]
    stage_fuel_mass = current_phase["stage_fuel_mass"]
    
    fuel_masses = []
    if isinstance(booster_fuel_mass, list):
        fuel_masses.extend(booster_fuel_mass)
    if isinstance(stage_fuel_mass, list):
        fuel_masses.extend(stage_fuel_mass)
    
    total_fuel_mass = np.sum(fuel_masses)
    
    # Calculate the mass flow rate
    massflow = total_fuel_mass / time_interval
    
    mdot = massflow * max_step
    dmdot = 0
    
    return massflow, mdot, dmdot

def calculate_atmosphere_data(date, lon, lat, h):
    # Current Position       # h in km required
    atmosphere = msis.run(np.datetime64(int(date), 's'), lon, lat, h/1000)
    rho = atmosphere[0, 0]
    temp = atmosphere[0, 10]
    p_a = rho * temp * R
    
    # Differential does not need to be specified since the data is manual calculated
    dp = 0
    drho = 0
    dT = 0
    
    return p_a, dp, rho, drho, temp, dT
    
def gravity_turn(h, g, v, gamma, scenario):
    h_1 = scenario["h_gt_start"] + scenario["alt"]
    h_2 = scenario["h_gt_end"] + scenario["alt"]
    gamma_gt_total = scenario["gamma_gt_total"] * np.pi/180     # rad

    if h < h_1:
        gamma_new = gamma
        diff_gamma = 0
        
    elif h >= h_1 and h <= h_2:
        # Example: in rad ->  90°  + (3/5  * -3°) = 88.2°
        gamma_new = gamma_0 + (((h - h_1)/( h_2 - h_1)) * -gamma_gt_total)
        diff_gamma = gamma_new - gamma
        
    elif h > h_2 and gamma > 0:
        gamma_new = gamma
        diff_gamma = -((g/v)-(v/(R_0+h)))*np.sin(gamma)
        #diff_gamma =  -(g/v)*np.cos(gamma)    
            
    else:
        gamma_new = math.radians(0)
        diff_gamma = 0
             
    return diff_gamma, gamma_new
#endregion

#region: ODE Functions
# ODE Definition
def trajectory_ODE(t,y,*ode_args):
    # 0.) Unpack ODE Arguments
    g_0, R_0, p_0, max_step, scenario, launch_vehicle, phases, phase_intervals = ode_args
    
    # 1.) Initial values of ODE-System (y = y_0 = [v_0, x_0, ...])
    v = y[0]
    x = y[1]
    s = y[2]
    h = y[3]
    gamma = y[4]
    m = y[5]
    date = y[6]
    lat = y[7]
    lon = y[8]
    p_a = y[9]
    rho = y[10]
    temp = y[11]
    F = y[12]
    mdot = y[13]
    
    # 2.) Definition of Phases and Phase Changes
    phase_changed, current_phase, prev_phase = get_phases(t, max_step, phases, phase_intervals)

    # 3.) Required Variables
    g = gravity(g_0, R_0, h)                        
    c_w = drag_coefficient(temp, v, rho, launch_vehicle)
    F_W = drag_force(c_w, rho, v, launch_vehicle)
    pressure_ratio = p_a / p_0
    F, y[12], dF = calculate_force(current_phase, pressure_ratio, h, scenario)
    if t == 0:
        massflow = y[13]  # Use the initial value passed in y
        dmdot = 0         # No derivative at t=0
    else:
        massflow, y[13], dmdot = calculate_massflow(current_phase)
    
    # 4.) Final ODE-System
    dv = ((F-F_W)/m - g*np.sin(gamma))                          # m/s
    dx = v * np.cos(gamma)                                      # m/s
    ds = v                                                      # m/s
    dh = v * np.sin(gamma)                                      # m/s
    diff_gamma, y[4] = gravity_turn(h, g, v, gamma, scenario) 
    if phase_changed:
        y[5] = prev_phase["mass_separation"]
        dm = 0
    else:
        dm = - massflow                                        # kg/s
    d_date = 1                                                  # 1 timestep
    y[7], dlat, y[8], dlon = coordinates(scenario["inclination"], scenario["lat"], scenario["lon"], x)
    y[9], dp, y[10], drho, y[11], dT = calculate_atmosphere_data(date, lon, lat, h)

    return [dv, dx, ds, dh, diff_gamma, dm, d_date , dlat, dlon, dp, drho, dT, dF, dmdot]

# Solving the ODE
def solve_ODE(y_0, ode_args):
    try:
        print(colors.BOLD + colors.BLUE + "Solving ODE..." + colors.END)
        
        # Event function for an ODE Break based on height
        def height_event(t, y, *args):
            h = y[3]
            return h - height_ode_break

        # Event properties
        height_event.terminal = True
        height_event.direction = 0
        
        # Solve ODE with or without Event Break based on height
        if use_ode_break:
            results = solve_ivp(
                trajectory_ODE, t_span, y_0, method=numeric_method,
                args=ode_args, max_step=max_step, atol=atol, rtol=rtol,
                events=height_event
            )
        else:
            results = solve_ivp(
                trajectory_ODE, t_span, y_0, method=numeric_method,
                args=ode_args, max_step=max_step, atol=atol, rtol=rtol
            )
        
        # Print Statements
        if results.status == 1:
            print(colors.BOLD + colors.GREEN + "ODE solution stopped as height reached the specified break point." + colors.END)
        else:
            print(colors.BOLD + colors.GREEN + "Solved ODE and returning data." + colors.END)
            
        return results
    
    except Exception as e:
        handle_error(e, "solve_ODE", "ODE could not be solved")  

#endregion