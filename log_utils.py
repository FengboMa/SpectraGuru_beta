import json
from filelock import FileLock
import os
import streamlit as st
import datetime as dt
import pandas as pd
from function_dict import names, references, reference_map
import numpy as np

count_log_file_path = "log/count_log.txt"
call_log_file_path = "log/call_log.txt"
user_log_file_path = "log/user_log.txt"


<<<<<<< HEAD
CULL_FUNCTION_TABLE_REFRESH = False # If false, function count table updates after any user interaction. If true, it only updates after a full browser refesh.
=======
references = {
    0:{"text":"Self-implemented", "link":False, "notes":None},
    1:{"text":"https://doi.org/10.1038/s41592-019-0686-2", "link":True, "notes":"SciPy reference"},
    2:{"text":"https://doi.org/10.1021/ac60214a047", "link":True, "notes":"Savitzky-Golay reference"},
    3:{"text":"https://doi.org/10.1039/b922045c", "link":True, "notes":"AirPLS reference"},
    4:{"text":"https://doi.org/10.1021/acs.analchem.5c01253", "link":True, "notes":"Optimized AirPLS reference"},
    5:{"text":"https://doi.org/10.1366/000370203322554518", "link":True, "notes":"TITLE: Automated Method for Subtraction of Fluorescence from Biological Raman Spectra"},
    6:{"text":"https://doi.org/10.1016/j.bios.2022.114721", "link":True, "notes":"TITLE: Rapid and quantitative detection of respiratory viruses using surface-enhanced Raman spectroscopy and machine learning"},
    7:{"text":"https://doi.org/10.1039/D2NR01277D", "link":True, "notes":"TITLE: Differentiation and classification of bacterial endotoxins based on surface enhanced Raman scattering and advanced machine learning"},
    8:{"text":"https://doi.org/10.1080/01621459.1963.10500845", "link":True, "notes":"Hierarchical clustering"},
    9:{"text":"https://doi.org/10.48550/arXiv.1201.0490", "link":True, "notes":"Scikit-learn reference"},
    10:{"text":"https://doi.org/10.1037/h0071325", "link":True, "notes":"PCA reference"},
    11:{"text":"https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf", "link":True, "notes":"t-SNE reference"}
}

reference_map = {
    'Processing_Despike_Auto':[0],
    'Processing_Despike_Manual':[0],
    'Processing_Smoothing_Savgol_Filter':[1,2],
    'Processing_Smoothing_FFT_Filter':[0],
    'Processing_Baseline_AirPLS':[3,4],
    'Processing_Baseline_Mod_Poly':[5],
    'Processing_Baseline_Gaussian_Lorentzian_Fitting':[6,7],
    'Processing_Normalization_Area':[0],
    'Processing_Normalization_Peak':[0],
    'Processing_Normalization_Minmax':[0],
    'Processing_Remove_Outliers':[0],
    'Analytics_Spectra_Derivation':[0],
    'Analytics_Correlation_Heatmap':[0],
    'Analytics_Peak_Identification':[1],
    'Analytics_Clustering_Clustermap':[1,8],
    'Analytics_Clustering_Dendrogram':[1,8],
    'Analytics_PCA':[9,10],
    'Analytics_TSNE':[9,11]
}

CULL_FUNCTION_TABLE_REFRESH = False # If false, function count table updates after any user interaction. If true, it only updates after a full browser refresh.
>>>>>>> 288ec7c (implements link-reference and multi-reference for function usage table; fills references based on SpectraGuru SI document)

def _ensure_parent_dir(file_path):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

# Creates a JSON String containing the details of a function call and appends the String to a file.
# function_name: the canonical name of the function
# function_params: a dictionary containing the parameters of interest used when calling the function
def log_function_call(f_name, f_params):

    call_number = increment_count(count_log_file_path, f_name)

    if st.session_state.get("user_logged_in", False) and st.session_state.get("user") and isinstance(st.session_state.user, dict):
        try:
            user = {
                "id":st.session_state.user['id'],
                "name":st.session_state.user['firstName'] + " " + st.session_state.user['lastName'],
                "email":st.session_state.user['email']
            }
        except (KeyError, TypeError):
            user = "Guest"
    else:
        user = "Guest"
    
    time = dt.datetime.now(dt.UTC).isoformat(timespec='milliseconds')

    entry = {
        "function_name":f_name,
        "call_number":call_number,
        "parameters":f_params,
        "user":user,
        "timestamp":time
    }

    json_str = json.dumps(entry, default=str)
    #print("JSON:", json_str)
    append_to_file(call_log_file_path, f"{json_str}\n")

# appends a specified string to a given file.
def append_to_file(file_path, string):
    _ensure_parent_dir(file_path)

    lock = FileLock(file_path + ".lock")

    with lock: # prevents two users from writing to the same file at once.
        with open(file_path, "a") as file:
            file.write(string)

# Returns a DataFrame used to display the call frequency of each of the processing/analysis functions.
def get_count_data():
    counts = {}

    lock = FileLock(count_log_file_path + ".lock")
    with lock:
        counts = read_counts_json(count_log_file_path)
    
    data = []
    num_functions = 0
    for key in counts.keys():
        readable_name = get_readable_name(key)
        reference = get_reference(key)
        entry = [readable_name['feature'], readable_name['algorithm'], counts[key], reference]
        data.append(entry)
        num_functions += 1
    
    data.sort(key=lambda entry: entry[2], reverse=True)
    
    df = pd.DataFrame(data=data, index=pd.RangeIndex(start=1, stop=num_functions+1), columns=['Feature', 'Algorithm', 'Usage (Times Called)', 'Reference'])
    return df

# Returns a user-readable name for a function given its keyname, in terms of both its feature name and algorithm name, if applicable.
def get_readable_name(keyname):

    readable_name = {}
    if keyname in names:
        readable_name = {
            'feature':names[keyname][0],
            'algorithm':names[keyname][1]
        }
    else:
        readable_name = {
            'feature':keyname,
            'algorithm':keyname
        }
    return readable_name

# Returns a string representing reference(s) to the literature for a given algorithm.
def get_reference(keyname):

    refs = ""

    if keyname in reference_map:
        ref_ids = reference_map[keyname]
        for i in range(len(ref_ids)):
            ref_id = ref_ids[i]
            text = references[ref_id]["text"]

            ref_string = ""
            # Decide whether the reference should be a link or raw text
            if references[ref_id]["link"]:

                if references[ref_id]["doc_page"]:
                    ref_string = f"[Docs]({text})"
                else:
                    ref_string = f"[Ref{i+1}]({text})"
            else:
                ref_string = text

            # Format as a comma-separated list
            if i > 0:
                refs += ", "
            refs += ref_string
    return refs


# Increments the counter for a specified metric in a given log file. Returns the new count and 
# creates a new entry if the keyname doesn't already exist.
def increment_count(file_path, keyname, amount=1):
    _ensure_parent_dir(file_path)
    lock = FileLock(file_path + ".lock")
    try:
        with lock:
            counts = read_counts_json(file_path)

            # if keyname doesn't exist, add it.
            if keyname not in counts:
                counts[keyname] = 0

            counts[keyname] += amount
            write_counts_json(file_path, counts)
        return counts[keyname]
    except Exception:
        return 0

# Returns a dictionary of all the key-value pairs expressed in a given log file. Log files must
# contain a single JSON String
def read_counts_json(file_path):
    
    counts = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                line = file.readline().strip()
                if line:
                    loaded = json.loads(line)
                    if isinstance(loaded, dict):
                        counts = loaded
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            counts = {}
                
    return counts

def write_counts_json(file_path, counts):
    _ensure_parent_dir(file_path)

    with open(file_path, "w") as file:
        file.write(json.dumps(counts))

# User count function
def log_user_count():
    return increment_count(user_log_file_path, 'Users')

# Plot_Generated count function
def log_plot_generated_count():
    return increment_count(user_log_file_path, 'Plots_Generated')

# Spectra_Processed count function
def log_spectra_processed_count():
    import streamlit as st
    return increment_count(user_log_file_path, 'Spectra_Processed', st.session_state.df[1:].shape[1])

# Essentially a function rename for clarity
def log_function_use_count(function_log_file_path, keyname, amount=1):
    return increment_count(function_log_file_path, keyname, amount)


### TEST FUNCTIONS ###
# The following functions are for convenience during testing and are not used for the application.

def clear_call_log():
    with open(call_log_file_path, "w") as file:
        file.write("")

def clear_count_log():
    write_counts_json(count_log_file_path, counts={})
