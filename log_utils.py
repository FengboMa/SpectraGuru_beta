import json

count_log_file_path = "log/count_log.txt"
call_log_file_path = "log/call_log.txt"
user_log_file_path = "log/user_log.txt"

# Creates a JSON String containing the details of a function call and appends the String to a file.
# function_name: the canonical name of the function
# function_params: a dictionary containing the parameters of interest used when calling the function
def log_function_call(function_name, function_params):

    call_number = read_counts()[function_name]
    increment_count(function_name)

    fc_entry = {
        "function_name":function_name,
        "call_number":call_number,
        "parameters":function_params
    }

    json_str = json.dumps(fc_entry)
    print("JSON:", json_str)

# Increments the counter for a specified metric in a given log file. Returns the new count and 
# returns 0 if the keyname does not match any recognizable keyname in the log file.
def increment_count(keyname, amount=1):
    try:
        counts = read_counts(count_log_file_path)
        counts[keyname] += amount
        write_counts(count_log_file_path, counts)
        return counts[keyname]
    except:
        return 0

# Returns a dictionary of all the key-value pairs expressed in a given log file. Log files must
# take the form:
#
# Key_1[\t]Value_1
# Key_2[\t]Value_2
# ...
def read_counts():
    import os

    counts = {}

    if os.path.exists(count_log_file_path):
        with open(count_log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts = {**counts, key: int(value)}
    
    return counts

# Writes a counts dictionary to a log file
def write_counts(counts):
    import os

    with open(count_log_file_path, "w") as file:
        for key, value in counts.items():
            file.write(f"{key}\t{value}\n")


# User count function
def log_user_count(log_file_path):
    return increment_count(log_file_path, 'User')

# Plot_Generated count function
def log_plot_generated_count(log_file_path):
    return increment_count(log_file_path, 'Plot_Generated')

# Spectra_Processed count function
def log_spectra_processed_count(log_file_path):
    import streamlit as st
    return increment_count(log_file_path, 'Spectra_Processed', st.session_state.df[1:].shape[1])

# Essentially a function rename for clarity
def log_function_use_count(function_log_file_path, keyname, amount=1):
    return increment_count(function_log_file_path, keyname, amount)