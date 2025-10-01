import streamlit as st
import pandas as pd
import function
import psycopg2
import numpy as np
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# Page-scoped key helper  ➜  every widget / cache key on this page is prefixed
# ─────────────────────────────────────────────────────────────────────────────
PAGE_ID = "upload"
def pkey(name: str) -> str:
    return f"{PAGE_ID}_{name}"

function.wide_space_default()
st.session_state.log_file_path = (
    r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group"
    r"\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software"
    r"\spectraApp_v14\element\user_count.txt"
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='utf-8')

@st.cache_data
def load_tab_data(file):
    return pd.read_csv(file, encoding='utf-8', sep='\t')

@st.cache_data
def load_multi_data(file_paths):
    def read_columns(file_path):
        try:
            df = pd.read_csv(
                file_path, delim_whitespace=True, skip_blank_lines=True,
                comment='#', header=None, on_bad_lines='skip'
            )
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] < 2:
                raise ValueError(f"File {file_path} does not contain at least two columns.")
            return df[0], df[1]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None

    first_columns, second_columns, file_names = [], [], []
    for file in file_paths:
        first_col, second_col = read_columns(file)
        if first_col is None or second_col is None:
            raise ValueError(f"Failed to read columns from file: {file}")
        first_columns.append(first_col)
        second_columns.append(second_col)
        file_names.append(file.name)
    if not first_columns:
        raise ValueError("No columns were read from the files.")
    reference_column = first_columns[0]
    for column in first_columns[1:]:
        if not reference_column.equals(column):
            raise ValueError("The first column is not the same in all files.")
    df = pd.concat(second_columns, axis=1)
    df.insert(0, 'RamanShift', reference_column)
    df.columns = ['RamanShift'] + file_names
    return df

def show_label_editor(temp_df, default_labels):
    """
    temp_df : DataFrame with columns ['RamanShift', spectra…]
    default_labels : dict{ spectrum_name : int }
    """
    st.write(
        """
        **Please use the following dataframe editor to insert labels**

        *Note:*  
        - The first column shows each spectrum column’s name.  
        - Edit the **Label** column to assign an integer class label.  
        - Double-click to edit; drag the cell’s bottom-right corner to copy.  
        - Only integer values are accepted.
        """
    )
    label_df = pd.DataFrame({
        'Spectrum': list(default_labels.keys()),
        'Label':    list(default_labels.values()),
        'Note':     ' '
    })
    column_config = {
        'Label': st.column_config.NumberColumn(
            label='Label',
            format='%d',
            step=1
        )
    }
    edited = st.data_editor(
        label_df,
        column_config=column_config,
        disabled=['Spectrum'],
        num_rows='fixed'
    )
    st.session_state.label_df = edited

# ----------------------------------------
st.write("## Data Upload")
st.divider()

# ----------------------------------------
# --- Automatic database connection ------------------------------------------
if 'db_logged_in' not in st.session_state:
    st.session_state.db_logged_in = False
if 'connection' not in st.session_state:
    try:
        conn = psycopg2.connect(
            dbname="SpectraGuruDB",
            user="sg_user",
            password="Aa123456",
            host="localhost",
            port="5432"
        )
        st.session_state.db_logged_in = True
        st.session_state.connection   = conn
        st.session_state.user         = "sg_user"
        st.session_state.passkey      = "Aa123456"
    except Exception as e:
        st.session_state.connection = None
        st.session_state.db_logged_in = False
        st.session_state.db_error     = str(e)

with st.expander("Database Login", expanded=False):
    if st.session_state.db_logged_in:
        st.success("Logged in as **sg_user**.")
    else:
        st.error(f"Database connection failed: {st.session_state.get('db_error', 'unknown error')}")

# ----------------------------------------
def reset_application():
    """
    Clears all keys in st.session_state except for 'run_count'.
    This is important to prevent the button from disappearing after a reset.
    """
    st.toast("Application has been reset!", icon="✅")
    
    # Keep track of the keys to delete
    keys_to_delete = []
    for key in st.session_state.keys():
        if key != 'run_count': # We want to preserve the run_count
            keys_to_delete.append(key)
    
    # Delete the keys
    for key in keys_to_delete:
        del st.session_state[key]

    # No need to call st.rerun() here, as the button click that
    # called this function already triggered a rerun.


# --- 2. Initialize and Increment the Counter ---

# Initialize the counter ONLY on the very first run.
if 'run_count' not in st.session_state:
    st.session_state.run_count = 0

# Increment the counter on EVERY script run.
st.session_state.run_count += 1


# --- 4. Conditionally Display the Button ---

# The button will only appear AFTER the first run.
if st.session_state.run_count > 1:
    st.button(
        "⚠️Reset Uploaded data and reset application",
        on_click=reset_application,
        type="primary"
    )


# --- Number of classes selector ---
n_classes = st.number_input(
    "Number of classes",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Set number of data classes (upload or query per class, in any combination)."
)
st.divider()

# === NEW: initialize global selection counters (total spectra across expanders)
if 'selected_counts' not in st.session_state:
    st.session_state.selected_counts = {}   # {class_idx: total_spectra_in_that_class}

# --- Data format options ---
FORMAT_OPTIONS = {
    "Multi TXT (two-column, common x)": {"kind": "multi_txt", "multi": True,  "types": ["txt"]},
    "Multi CSV (two-column, common x)": {"kind": "multi_csv", "multi": True,  "types": ["csv"]},
    "Single TSV (.tsv / tab-separated)": {"kind": "single_tsv", "multi": False, "types": ["csv"]},
    "Single CSV (comma-separated)":      {"kind": "single_csv", "multi": False, "types": ["csv"]},
}
def process_upload(kind, uploaded):
    if kind in ("multi_txt", "multi_csv"):
        return load_multi_data(uploaded) if uploaded else None
    if kind == "single_tsv":
        return load_tab_data(uploaded) if uploaded is not None else None
    if kind == "single_csv":
        return load_data(uploaded) if uploaded is not None else None
    return None

if n_classes > 1:
    st.warning(
        "All uploaded classes will be **automatically interpolated** to a "
        "common 1 cm⁻¹ Raman-shift grid and then **cropped** to the shared "
        "overlap range before further analysis. "
        "This ensures every spectrum has identical x-axis values."
    )

# --- Main per-class input ----------------------------------------------------
class_dfs    = []
class_labels = []
all_ready    = True

for idx in range(int(n_classes)):
    with st.expander(f"Class {idx + 1}", expanded=True):

        # 0. Class label --------------------------------------------------
        class_label = st.text_input(
            "Class label (optional, used in column names)",
            value=f"Class_{idx + 1}",
            key=pkey(f"class_label_{idx}")
        )

        # 1. Restore cached data -----------------------------------------
        cache_key = pkey(f"class_df_{idx}")
        df_cached = st.session_state.get(cache_key)
        if df_cached is not None:
            st.success(f"{class_label} already loaded. Shape: {df_cached.shape}")
            st.dataframe(df_cached.head())

            # NEW: count spectra columns for cached data and store for this class
            manual_count = max(0, df_cached.shape[1] - 1)  # minus RamanShift
            st.session_state.selected_counts[idx] = manual_count
            total_selected_preview = sum(st.session_state.selected_counts.values())
            st.caption(f"This class: **{manual_count}** spectra · Total: **{total_selected_preview} / 1000**")

            class_labels.append(class_label)
            class_dfs.append((class_label, df_cached))
            continue

        # 2. Data source --------------------------------------------------
        source = st.radio(
            "Data Source", ["Manual Upload", "Database Query"],
            horizontal=True, key=pkey(f"source_{idx}")
        )
        df_this = None

        # ======================= 2-A. MANUAL UPLOAD ======================
        if source == "Manual Upload":
            fmt = st.selectbox(
                "Select file format",
                list(FORMAT_OPTIONS.keys()),
                index=None,
                placeholder="Choose a format ...",
                key=pkey(f"class_fmt_{idx}")
            )
            if fmt:
                cfg = FORMAT_OPTIONS[fmt]
                if cfg["multi"]:
                    uploads = st.file_uploader(
                        "Upload file(s)",
                        type=cfg["types"],
                        accept_multiple_files=True,
                        key=pkey(f"class_upload_{idx}")
                    )
                    if uploads:
                        try:
                            df_this = process_upload(cfg["kind"], uploads)
                        except Exception as e:
                            st.error(f"Error loading {class_label}: {e}")
                            all_ready = False
                    else:
                        all_ready = False
                else:
                    upload = st.file_uploader(
                        "Upload file",
                        type=cfg["types"],
                        key=pkey(f"class_upload_{idx}")
                    )
                    if upload:
                        try:
                            df_this = process_upload(cfg["kind"], upload)
                        except Exception as e:
                            st.error(f"Error loading {class_label}: {e}")
                            all_ready = False
                    else:
                        all_ready = False
            else:
                all_ready = False

            # Sample data option for single-class upload
            if n_classes == 1 and (df_this is None or not fmt):
                st.info("Don't have data? Try with sample data.")
                if st.button("Load sample data", key=pkey("sample_data_btn")):
                    try:
                        df_sample = load_data(r'element/sampledata.csv')
                        df_this   = df_sample
                        st.session_state.df         = df_sample
                        st.session_state.backup     = df_sample
                        st.session_state.class_data = [("Sample", df_sample)]
                        st.success("Sample data loaded.")
                    except Exception as e:
                        st.error(f"Failed to load sample data: {e}")

            if df_this is not None:
                st.success(f"{class_label} loaded. Shape: {df_this.shape}")
                st.dataframe(df_this.head())
                st.session_state[cache_key] = df_this

                # NEW: manual uploads contribute to the global total
                manual_count = max(0, df_this.shape[1] - 1)  # minus RamanShift
                st.session_state.selected_counts[idx] = manual_count
                prospective_total = sum(st.session_state.selected_counts.values())
                st.caption(f"This class: **{manual_count}** spectra · Total (all classes): **{prospective_total} / 1000**")
                if prospective_total > 1000:
                    st.error("Total selected spectra exceed **1000**. Remove data to proceed.")
                    all_ready = False
                elif prospective_total > 500:
                    st.warning("Total selected spectra exceed **500**. Processing may be slow.")
            else:
                st.info("Waiting for format & upload." if fmt else "Select a format.")
                # ensure this class doesn't count yet
                st.session_state.selected_counts[idx] = 0

        # ======================= 2-B. DATABASE QUERY ======================
        else:
            if not st.session_state.db_logged_in:
                st.warning("Please log in to the database above before querying.")
                all_ready = False
                st.session_state.selected_counts[idx] = 0
            else:
                advanced_search = st.checkbox("Advanced Search", key=pkey(f"adv_{idx}"))
                data_type_filter = (
                    st.radio(
                        "Select Data Type to Search:",
                        ["Both", "Raw Data Only", "Standard Data Only"], index=0,
                        horizontal=True, key=pkey(f"dtf_{idx}")
                    )
                    if advanced_search else "Both"
                )

                search_query = st.text_input(
                    "Search keyword (database):", key=pkey(f"search_{idx}")
                )
                if search_query:
                    results = function.search_database(search_query, data_type_filter)
                    if not results.empty:
                        results['Select'] = False
                        cols = ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type', 'batch_spectrum_count', 'batch_concentration', 'batch_concentration_units']
                        results = results[cols + [c for c in results.columns if c not in cols]]
                        edited_results = st.data_editor(
                            results,
                            column_config={"Select": st.column_config.CheckboxColumn()}
                        )
                        selected_rows = edited_results[edited_results['Select']]

                        # NEW: compute spectra count for this class from DB selection
                        if 'batch_spectrum_count' in selected_rows.columns:
                            class_sel_count = int(selected_rows['batch_spectrum_count'].fillna(0).sum())
                        else:
                            class_sel_count = int(len(selected_rows))

                        # Compute the prospective global total (include this class’s current selection)
                        other_total = sum(v for k, v in st.session_state.selected_counts.items() if k != idx)
                        prospective_total = other_total + class_sel_count

                        # Store/update this class’s current selection count
                        st.session_state.selected_counts[idx] = class_sel_count

                        if class_sel_count == 0:
                            st.info("Select at least one batch for this class.")
                            all_ready = False
                        else:
                            sel_preview = ", ".join(f"{r['batch_analyte_name']}_{r['batch_id']}" for _, r in selected_rows.iterrows())
                            st.write(f"Selected: {sel_preview}")
                            st.caption(f"This class: **{class_sel_count}** spectra · Total (all classes): **{prospective_total} / 1000**")

                            # Global safeguards
                            if prospective_total > 1000:
                                st.error("Total selected spectra exceed **1000**. Reduce your selection to enable **Get Data**.")
                                allow_get = False
                                all_ready = False
                            else:
                                if prospective_total > 500:
                                    st.warning("Total selected spectra exceed **500**. Processing may be slow.")
                                allow_get = True

                            # Only show the button if allowed under the global rule
                            if allow_get and st.button("Get Data", key=pkey(f"getdata_{idx}")):
                                with st.spinner("Fetching and processing spectrum data..."):
                                    try:
                                        conn = st.session_state.connection
                                        data_dfs, mins, maxs = [], [], []

                                        for _, sel in selected_rows.iterrows():
                                            cur = conn.cursor()
                                            batch_id   = sel['batch_id']
                                            batch_type = sel['batch_data_type'].lower()

                                            if batch_type == 'raw':
                                                query = """
                                                    SELECT sd.* FROM spectrum s
                                                    JOIN spectrum_data sd ON s.spectrum_id = sd.spectrum_id
                                                    WHERE s.batch_id = %s
                                                """
                                            else:  # standard
                                            # NOTE: use the standard tables
                                                query = """
                                                    SELECT sds.* FROM spectrum_standard ss
                                                    JOIN spectrum_data_standard sds
                                                    ON ss.spectrum_standard_id = sds.spectrum_standard_id
                                                    WHERE ss.batch_standard_id = %s
                                                """

                                            cur.execute(query, (int(batch_id),))
                                            col_names = [d[0] for d in cur.description]
                                            data      = cur.fetchall()
                                            cur.close()

                                            if not data:
                                                st.warning(f"No spectrum data in batch {batch_id}.")
                                                continue

                                            df_raw          = pd.DataFrame(data, columns=col_names)
                                            spectrum_id_col = df_raw.columns[1]
                                            prefix = f"{sel['batch_analyte_name']}_{batch_id}_"
                                            df_raw['spectrum_column'] = prefix + df_raw[spectrum_id_col].astype(str)

                                            pivot_df = (
                                                df_raw.pivot_table(
                                                    index='wavenumber',
                                                    columns='spectrum_column',
                                                    values='intensity'
                                                )
                                                .reset_index()
                                                .rename(columns={'wavenumber': 'RamanShift'})
                                            )

                                            data_dfs.append(pivot_df)
                                            x_vals = pd.to_numeric(pivot_df['RamanShift'])
                                            mins.append(x_vals.min())
                                            maxs.append(x_vals.max())

                                        if not data_dfs:
                                            st.warning("No usable spectra fetched.")
                                            all_ready = False
                                        else:
                                            common_min = int(np.ceil(max(mins)))
                                            common_max = int(np.floor(min(maxs)))
                                            if common_min >= common_max:
                                                st.error("Selected batches have no overlapping Raman-shift range.")
                                                all_ready = False
                                            else:
                                                ref_x = np.arange(common_min, common_max + 1, 1)
                                                interp_dfs = []
                                                for df_tmp in data_dfs:
                                                    x_orig = pd.to_numeric(df_tmp['RamanShift'])
                                                    interp_df = pd.DataFrame({'RamanShift': ref_x})
                                                    for col in df_tmp.columns[1:]:
                                                        f = interp1d(
                                                            x_orig,
                                                            pd.to_numeric(df_tmp[col], errors='coerce'),
                                                            kind='linear',
                                                            bounds_error=False,
                                                            fill_value="extrapolate"
                                                        )
                                                        interp_df[col] = f(ref_x)
                                                    interp_dfs.append(interp_df.set_index('RamanShift'))
                                                df_this = (
                                                    pd.concat(interp_dfs, axis=1)
                                                    .reset_index().drop_duplicates()
                                                )
                                                st.success(f"Fetched {len(data_dfs)} batch(es) for {class_label}.")
                                                st.dataframe(df_this.head())
                                                st.session_state[cache_key] = df_this
                                    except Exception as e:
                                        st.error(f"Error fetching data: {e}")
                                        all_ready = False
                    else:
                        st.warning("No search results found.")
                        all_ready = False
                        st.session_state.selected_counts[idx] = 0
                else:
                    st.info("Enter a keyword to search the database.")
                    all_ready = False
                    st.session_state.selected_counts[idx] = 0

        # 3. Book-keeping -------------------------------------------------
        class_labels.append(class_label)
        class_dfs.append((class_label, df_this))
        if df_this is None:
            all_ready = False

# --- Global selection summary (after loop) -----------------------------------
total_selected = sum(st.session_state.selected_counts.values()) if 'selected_counts' in st.session_state else 0
st.markdown(f"**Total selected spectra across all classes:** {total_selected} / 1000")
if total_selected > 1000:
    st.error("Reduce your total below **1000** to proceed.")
elif total_selected > 500:
    st.warning("Total exceeds **500**; processing may be slow.")

# --- Interpolate and combine classes ----------------------------------------
# Guard against proceeding if total over 1000
if total_selected > 1000:
    all_ready = False

if all_ready and all(df is not None for _, df in class_dfs):
    st.success(f"All {n_classes} classes loaded successfully.")
    mins, maxs = [], []
    for _, df in class_dfs:
        x = pd.to_numeric(df.iloc[:, 0])
        mins.append(x.min())
        maxs.append(x.max())
    common_min = int(np.ceil(max(mins)))
    common_max = int(np.floor(min(maxs)))
    if common_min >= common_max:
        st.error("Uploaded/selected spectra have no overlapping Raman-shift range.")
        st.stop()
    ref_x = np.arange(common_min, common_max + 1, 1)
    st.session_state.interpolation_ref_x = ref_x

    interpolated_class_dfs = []
    for class_label, df in class_dfs:
        x_orig = pd.to_numeric(df.iloc[:, 0])
        interp_df = pd.DataFrame({'RamanShift': ref_x})
        for col in df.columns[1:]:
            f = interp1d(
                x_orig,
                pd.to_numeric(df[col], errors='coerce'),
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            interp_df[col] = f(ref_x)
        rename_map = {c: f"{class_label}__{c}" for c in interp_df.columns[1:]}
        interp_df.rename(columns=rename_map, inplace=True)
        interpolated_class_dfs.append((class_label, interp_df))

    combined_df = (
        pd.concat([df.set_index('RamanShift') for _, df in interpolated_class_dfs], axis=1)
        .reset_index().drop_duplicates()
    )
    st.session_state.class_data = interpolated_class_dfs
    st.session_state.df        = combined_df
    st.session_state.backup    = combined_df.copy() 

    # --- Label editor ---
    default_lbls_multi = {}
    for idx, (class_label, df_cls) in enumerate(interpolated_class_dfs, start=1):
        for col in df_cls.columns[1:]:
            default_lbls_multi[col] = idx
    show_label_editor(combined_df, default_lbls_multi)

elif all_ready and all(df is not None for _, df in class_dfs) and n_classes == 1:
    _, df = class_dfs[0]
    st.session_state.df     = df
    st.session_state.backup = df.copy() 
    st.session_state.class_data = class_dfs
    default_lbls_single = {col: 1 for col in df.columns[1:]}
    show_label_editor(df, default_lbls_single)

else:
    st.warning("Waiting for all classes to be ready. Make sure each class is fully uploaded or queried.")

# --- Downstream navigation/preview ------------------------------------------
if 'df' in st.session_state:
    st.divider()
    col1, col2 = st.columns([2, 12])
    if col1.button(label='Processing Page', key=pkey('switch_processing_page')):
        st.switch_page("pages/3_Processing.py")
    col2.markdown(' :arrow_left: **Go to Processing Page to process data**')
    st.write("#### Preview")
    st.write(st.session_state.backup)
    function.update_mode_option()
