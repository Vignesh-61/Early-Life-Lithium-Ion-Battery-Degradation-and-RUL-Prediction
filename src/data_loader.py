"""
data_loader.py — Load NASA Battery .mat files
================================================

What this module does:
    Loads the NASA Li-ion Battery Aging Dataset from MATLAB .mat files.
    Each .mat file contains a nested structure with charge, discharge,
    and impedance cycles for one battery.

Why we need it:
    The raw data is in MATLAB format with deeply nested arrays.
    This module handles all the complexity of parsing the MATLAB
    structure so that the rest of the project works with clean
    Python data structures.

How it works:
    1. Uses scipy.io.loadmat() to read .mat files
    2. Navigates the nested MATLAB structure
    3. Extracts cycle-by-cycle measurements
    4. Returns organized Python dictionaries

What happens if we don't use it:
    Every notebook and script would need to duplicate the complex
    .mat parsing logic, leading to bugs and inconsistencies.
"""

import os
import numpy as np
import scipy.io as sio


def load_battery_mat(filepath):
    """
    Load a single NASA battery .mat file and extract all cycles.
    
    Parameters
    ----------
    filepath : str
        Path to the .mat file (e.g., 'data/raw/B0005.mat')
    
    Returns
    -------
    dict with keys:
        'battery_id': str — battery identifier (e.g., 'B0005')
        'cycles': list of dicts — one dict per cycle with fields:
            'cycle_index': int — index in the original data
            'type': str — 'charge', 'discharge', or 'impedance'
            'ambient_temperature': float — ambient temp in °C
            'data': dict — measurement arrays (varies by cycle type)
    """
    # Extract battery ID from filename
    battery_id = os.path.splitext(os.path.basename(filepath))[0]
    
    # Load the MATLAB file
    # scipy.io.loadmat returns a dictionary with the variable names as keys
    mat_data = sio.loadmat(filepath)
    
    # The battery data is stored under the battery ID key
    # Structure: mat_data[battery_id] -> shape (1,1) -> ['cycle'] -> shape (1, N_cycles)
    battery_struct = mat_data[battery_id]
    cycle_array = battery_struct['cycle'][0, 0]
    n_cycles = cycle_array.shape[1]
    
    print(f"[data_loader] Loading {battery_id}: {n_cycles} total cycles found")
    
    cycles = []
    for i in range(n_cycles):
        cycle_struct = cycle_array[0, i]
        
        # Extract cycle type (charge, discharge, or impedance)
        cycle_type = str(cycle_struct['type'][0])
        
        # Extract ambient temperature
        ambient_temp = float(cycle_struct['ambient_temperature'][0, 0])
        
        # Extract measurement data based on cycle type
        data_struct = cycle_struct['data'][0, 0]
        data = {}
        
        if data_struct.dtype.names:
            for field_name in data_struct.dtype.names:
                try:
                    value = data_struct[field_name]
                    # Flatten single-row arrays to 1D
                    if isinstance(value, np.ndarray):
                        if value.ndim == 2 and value.shape[0] == 1:
                            value = value.flatten()
                        elif value.ndim == 2 and value.shape == (1, 1):
                            value = float(value[0, 0])
                    data[field_name] = value
                except Exception as e:
                    print(f"  Warning: Could not read field '{field_name}' "
                          f"in cycle {i}: {e}")
        
        cycles.append({
            'cycle_index': i,
            'type': cycle_type,
            'ambient_temperature': ambient_temp,
            'data': data
        })
    
    # Print summary
    type_counts = {}
    for c in cycles:
        type_counts[c['type']] = type_counts.get(c['type'], 0) + 1
    print(f"[data_loader] {battery_id} cycle types: {type_counts}")
    
    return {
        'battery_id': battery_id,
        'cycles': cycles
    }


def load_all_batteries(data_dir):
    """
    Load all battery .mat files from a directory.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing .mat files
    
    Returns
    -------
    dict : battery_id -> battery data dict
    """
    batteries = {}
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir}. "
            f"Please ensure the NASA battery data files are present."
        )
    
    print(f"[data_loader] Found {len(mat_files)} battery files: {mat_files}")
    
    for filename in mat_files:
        filepath = os.path.join(data_dir, filename)
        battery_data = load_battery_mat(filepath)
        batteries[battery_data['battery_id']] = battery_data
    
    return batteries


def extract_discharge_cycles(battery_data):
    """
    Extract only discharge cycles from a battery's data.
    
    Why discharge cycles?
        Discharge cycles contain the 'Capacity' measurement, which is
        our primary prediction target. Capacity is measured as the total
        charge delivered during discharge down to the cutoff voltage.
    
    Parameters
    ----------
    battery_data : dict
        Output from load_battery_mat()
    
    Returns
    -------
    list of dicts — only discharge cycles, with discharge_number added
    """
    discharge_cycles = []
    discharge_num = 0
    
    for cycle in battery_data['cycles']:
        if cycle['type'] == 'discharge':
            cycle_copy = cycle.copy()
            cycle_copy['discharge_number'] = discharge_num
            discharge_cycles.append(cycle_copy)
            discharge_num += 1
    
    print(f"[data_loader] {battery_data['battery_id']}: "
          f"{len(discharge_cycles)} discharge cycles extracted")
    
    return discharge_cycles


def get_discharge_capacities(battery_data):
    """
    Extract the sequence of discharge capacities for a battery.
    
    This gives us the degradation curve — the main signal we want
    our DNN to predict.
    
    Parameters
    ----------
    battery_data : dict
        Output from load_battery_mat()
    
    Returns
    -------
    discharge_numbers : list of int
    capacities : list of float — capacity in Ah for each discharge cycle
    """
    discharge_numbers = []
    capacities = []
    discharge_num = 0
    
    for cycle in battery_data['cycles']:
        if cycle['type'] == 'discharge':
            data = cycle['data']
            if 'Capacity' in data:
                cap = data['Capacity']
                if isinstance(cap, np.ndarray):
                    cap = float(np.max(cap))
                else:
                    cap = float(cap)
                discharge_numbers.append(discharge_num)
                capacities.append(cap)
            discharge_num += 1
    
    return discharge_numbers, capacities


# ---- Quick test ----
if __name__ == '__main__':
    import sys
    
    # Default path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    batteries = load_all_batteries(data_dir)
    
    print("\n" + "=" * 60)
    print("BATTERY SUMMARY")
    print("=" * 60)
    
    for bat_id, bat_data in sorted(batteries.items()):
        nums, caps = get_discharge_capacities(bat_data)
        if caps:
            print(f"\n{bat_id}:")
            print(f"  Discharge cycles: {len(caps)}")
            print(f"  Initial capacity: {caps[0]:.4f} Ah")
            print(f"  Final capacity:   {caps[-1]:.4f} Ah")
            print(f"  Capacity fade:    {(1 - caps[-1]/caps[0])*100:.1f}%")
