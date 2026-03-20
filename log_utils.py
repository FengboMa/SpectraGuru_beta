import json
from filelock import FileLock
import os
import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np

count_log_file_path = "log/count_log.txt"
call_log_file_path = "log/call_log.txt"
user_log_file_path = "log/user_log.txt"

# Creates a JSON String containing the details of a function call and appends the String to a file.
# function_name: the canonical name of the function
# function_params: a dictionary containing the parameters of interest used when calling the function
def log_function_call(f_name, f_params):

    increment_count(count_log_file_path, f_name)
    call_number = read_counts_json(count_log_file_path)[f_name]

    if st.session_state.user_logged_in:
        #print(st.session_state.user)
        user = {
            "id":st.session_state.user['id'],
            "name":st.session_state.user['firstName'] + " " + st.session_state.user['lastName'],
            "email":st.session_state.user['email']
        }
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

    json_str = json.dumps(entry)
    #print("JSON:", json_str)
    append_to_file(call_log_file_path, f"{json_str}\n")

# appends a specified string to a given file.
def append_to_file(file_path, string):

    lock = FileLock(file_path + ".lock")

    if os.path.exists(file_path):
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
    
    names = {
        'Processing_Despike_Auto':("Despike", "Automatic Despike"),
        'Processing_Despike_Manual':("Despike", "Manual Despike"),
        'Processing_Smoothing_Savgol_Filter':("Smoothening", "Savitzky-Golay filter"),
        'Processing_Smoothing_FFT_Filter':("Smoothening", "1D Fast Fourier Transform filter"),
        'Processing_Baseline_AirPLS':("Baseline Removal", "AirPLS"),
        'Processing_Baseline_Mod_Poly':("Basline Removal", "Modified Polynomial Fitting"),
        'Processing_Baseline_Gaussian_Lorentzian_Fitting':("Baseline Removal", "Gaussian-Lorentzian Fitting"),
        'Processing_Normalization_Area':("Normalization", "Normalization by Area"),
        'Processing_Normalization_Peak':("Normalizarion", "Normalization by Peak"),
        'Processing_Normalization_Minmax':("Normalization", "Min-max normalization"),
        'Processing_Remove_Outliers':("Outlier Removal", "Outlier Removal"),
        'Analytics_Spectra_Derivation':("Spectra Derivation", "Spectra Derivation"),
        'Analytics_Correlation_Heatmap':("Correlation Heatmap", "Correlation Heatmap"),
        'Analytics_Peak_Identification':("Peak Identification", "Peak Identification"),
        'Analytics_Clustering_Clustermap':("Hierarchical Clustering", "Clustermap"),
        'Analytics_Clustering_Dendrogram':("Hierarchical Clustering", "Dendrogram"),
        'Analytics_PCA':("PCA", "Principal Component Analysis"),
        'Analytics_TSNE':("t-SNE", "t-Distributed Stochastic Neighbor Embedding")
    }

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

def get_reference(keyname):
    return ""

# Increments the counter for a specified metric in a given log file. Returns the new count and 
# creates a new entry if the keyname doesn't already exist.
def increment_count(file_path, keyname, amount=1):
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
        with open(file_path, "r") as file:
            counts = json.loads(file.readline())
                
    return counts

def write_counts_json(file_path, counts):

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
