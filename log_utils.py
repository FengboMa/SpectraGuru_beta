import json
from filelock import FileLock
import os
import streamlit as st
import datetime as dt
import pandas as pd
from urllib.parse import quote
from function_dict import names, references, reference_map

count_log_file_path = "log/count_log.txt"
call_log_file_path = "log/call_log.txt"
user_log_file_path = "log/user_log.txt"

CULL_FUNCTION_TABLE_REFRESH = False # If false, function count table updates after any user interaction. If true, it only updates after a full browser refesh.

PAGE_BADGES = {
    "Processing": ":blue-badge[Processing]",
    "Analytics": ":green-badge[Analytics]",
    "Toolbox": ":orange-badge[Toolbox]",
}

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
        page = get_page_badge(key)
        references_text = get_reference(key)
        entry = [page, readable_name['algorithm'], counts[key], references_text]
        data.append(entry)
        num_functions += 1
    
    data.sort(key=lambda entry: entry[2], reverse=True)
    
    df = pd.DataFrame(
        data=data,
        index=pd.RangeIndex(start=1, stop=num_functions+1),
        columns=['Page', 'Algorithm', 'Usage Count', 'Documentation and References']
    )
    return df


def get_page_badge(keyname):
    page = keyname.split("_", 1)[0]
    return PAGE_BADGES.get(page, page)


def format_markdown_link(label, url):
    safe_url = quote(url, safe=":/?#[]@!$&'*,;=%~+-._")
    return f"[{label}]({safe_url})"

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

# Returns Streamlit Markdown for documentation and literature links for a given algorithm.
def get_reference(keyname):

    refs = []

    if keyname in reference_map:
        ref_ids = reference_map[keyname]
        for ref_id in ref_ids:
            text = references[ref_id]["text"]

            if not text or not references[ref_id]["link"]:
                continue

            if references[ref_id]["doc_page"]:
                label = "Documentation"
            else:
                label = "Reference"
            refs.append(format_markdown_link(label, text))
    return ", ".join(refs)

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
