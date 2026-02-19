import json
from filelock import FileLock
import os
import streamlit as st

count_log_file_path = "log/count_log.txt"
call_log_file_path = "log/call_log.txt"
user_log_file_path = "log/user_log.txt"

# Creates a JSON String containing the details of a function call and appends the String to a file.
# function_name: the canonical name of the function
# function_params: a dictionary containing the parameters of interest used when calling the function
def log_function_call(f_name, f_params):

    increment_count(count_log_file_path, f_name)
    call_number = read_counts_json(count_log_file_path)[f_name]

    entry = {
        "function_name":f_name,
        "call_number":call_number,
        "parameters":f_params,
        "user":st.session_state.user.email
    }

    json_str = json.dumps(entry)
    print("JSON:", json_str)
    append_to_file(call_log_file_path, f"{json_str}\n")

# appends a specified string to a given file.
def append_to_file(file_path, string):

    lock = FileLock(file_path + ".lock")

    if os.path.exists(file_path):
        with lock: # prevents two users from writing to the same file at once.
            with open(file_path, "a") as file:
                file.write(string)

# Increments the counter for a specified metric in a given log file. Returns the new count and 
# returns 0 if the keyname does not match any recognizable keyname in the log file.
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
