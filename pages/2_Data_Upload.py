import streamlit as st
import pandas as pd
import function
import psycopg2
import numpy as np
from scipy.interpolate import interp1d

function.wide_space_default()
st.session_state.log_file_path = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\user_count.txt"

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
            df = pd.read_csv(file_path, delim_whitespace=True, skip_blank_lines=True,
                             comment='#', header=None, on_bad_lines='skip')
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
        - Double‑click to edit; drag the cell’s bottom‑right corner to copy.  
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
        disabled=['Spectrum'],      # make 1st column read‑only
        num_rows='fixed'
    )
    st.session_state.label_df = edited

# ----------------------------------------
st.write("## Data Upload")
st.divider()

# ----------------------------------------
# --- Database Login Panel (at the top) ---
if 'db_logged_in' not in st.session_state:
    st.session_state.db_logged_in = False
if 'connection' not in st.session_state:
    st.session_state.connection = None

with st.expander("Database Login (for Database Query option, optional)", expanded=False):
    db_user = st.text_input("User", value=st.session_state.get('user', ''))
    db_pass = st.text_input("Passkey", type="password", value=st.session_state.get('passkey', ''))
    if st.button("Log in to Database"):
        try:
            conn = psycopg2.connect(
                dbname="SpectraGuruDB",
                user=db_user,
                password=db_pass,
                host="localhost",
                port="5432"
            )
            st.session_state.connection = conn
            st.session_state.db_logged_in = True
            st.session_state.user = db_user
            st.session_state.passkey = db_pass
            st.success("Database connection successful!")
        except Exception as e:
            st.session_state.db_logged_in = False
            st.error(f"Database connection error: {e}")

# ----------------------------------------
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

# --------- Multi-class warning ---------
if n_classes > 1:
    st.warning(
        "All uploaded classes will be **automatically interpolated** to a "
        "common 1 cm⁻¹ Raman‑shift grid and then **cropped** to the shared "
        "overlap range before further analysis. "
        "This ensures every spectrum has identical x‑axis values."
    )

# --- Main per-class input ---
class_dfs = []
class_labels = []
all_ready = True

for idx in range(int(n_classes)):
    with st.expander(f"Class {idx + 1}", expanded=True):
        # 1. Choose data source for this class
        source = st.radio(
            "Data Source", ["Manual Upload", "Database Query"], horizontal=True, key=f"source_{idx}"
        )
        class_label = st.text_input(
            "Class label (optional, used in column names)",
            value=f"Class_{idx + 1}", key=f"class_label_{idx}"
        )
        df_this = None

        if source == "Manual Upload":
            fmt = st.selectbox(
                "Select file format",
                list(FORMAT_OPTIONS.keys()),
                index=None,
                placeholder="Choose a format ...",
                key=f"class_fmt_{idx}"
            )
            if fmt:
                cfg = FORMAT_OPTIONS[fmt]
                if cfg["multi"]:
                    uploads = st.file_uploader(
                        "Upload file(s)",
                        type=cfg["types"],
                        accept_multiple_files=True,
                        key=f"class_upload_{idx}"
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
                        key=f"class_upload_{idx}"
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
            # ---- Sample data option for single class manual upload ----
            if n_classes == 1 and (df_this is None or not fmt):
                st.info("Don't have data? Try with sample data.")
                if st.button("Load sample data", key="sample_data_btn"):
                    try:
                        df_sample = load_data(r'element/sampledata.csv')
                        df_this = df_sample
                        st.session_state.df = df_sample
                        st.session_state.backup = df_sample
                        st.session_state.class_data = [("Sample", df_sample)]
                        st.success("Sample data loaded.")
                        # st.dataframe(df_sample.head())
                    except Exception as e:
                        st.error(f"Failed to load sample data: {e}")
            if df_this is not None:
                st.success(f"{class_label} loaded. Shape: {df_this.shape}")
                st.dataframe(df_this.head())
            else:
                st.info("Waiting for format & upload." if fmt else "Select a format.")

        else:  # Database Query (FORCE SINGLE SELECT)
            if not st.session_state.db_logged_in:
                st.warning("Please log in to the database above before querying.")
                all_ready = False
            else:
                advanced_search = st.checkbox("Advanced Search", key=f"adv_{idx}")
                if advanced_search:
                    data_type_filter = st.radio(
                        "Select Data Type to Search:",
                        ["Both", "Raw Data Only", "Standard Data Only"], index=0, horizontal=True,
                        key=f"dtf_{idx}"
                    )
                else:
                    data_type_filter = "Both"
                search_query = st.text_input("Search keyword (database):", key=f"search_{idx}")
                if search_query:
                    results = function.search_database(search_query, data_type_filter)
                    if not results.empty:
                        results['Select'] = False
                        columns_order = ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type'] + \
                                        [col for col in results.columns if col not in ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type']]
                        results = results[columns_order]
                        edited_results = st.data_editor(
                            results, column_config={"Select": st.column_config.CheckboxColumn()}
                        )
                        selected_rows = edited_results[edited_results['Select'] == True]
                        # FORCE ONLY ONE SELECTION!
                        if len(selected_rows) > 1:
                            st.error("Please select **only one** record for this class.")
                            all_ready = False
                        elif not selected_rows.empty:
                            selected_row = selected_rows.iloc[0]
                            st.write(f"Selected: {selected_row['batch_analyte_name']}_{selected_row['batch_data_type']}_{selected_row['batch_id']}")
                            if st.button("Get Data", key=f"getdata_{idx}"):
                                with st.spinner("Fetching and processing spectrum data..."):
                                    try:
                                        conn = st.session_state.connection
                                        cur = conn.cursor()
                                        batch_id = selected_row['batch_id']
                                        batch_data_type = selected_row['batch_data_type'].lower()
                                        if batch_data_type == 'raw':
                                            query = """
                                                SELECT sd.* FROM spectrum s
                                                JOIN spectrum_data sd ON s.spectrum_id = sd.spectrum_id
                                                WHERE s.batch_id = %s
                                            """
                                        elif batch_data_type == 'standard':
                                            query = """
                                                SELECT sds.* FROM spectrum_standard ss
                                                JOIN spectrum_data_standard sds ON ss.spectrum_standard_id = sds.spectrum_standard_id
                                                WHERE ss.batch_standard_id = %s
                                            """
                                        else:
                                            st.warning(f"Unknown data type: {batch_data_type}")
                                            cur.close()
                                            all_ready = False
                                            continue
                                        cur.execute(query, (int(batch_id),))
                                        data = cur.fetchall()
                                        if not data:
                                            st.warning("No spectrum data fetched.")
                                            all_ready = False
                                        else:
                                            col_names = [desc[0] for desc in cur.description]
                                            df = pd.DataFrame(data, columns=col_names)
                                            spectrum_id_col = df.columns[1]
                                            unique_label = (
                                                f"{selected_row['batch_analyte_name']}_"
                                                f"{selected_row['batch_data_type']}_"
                                                f"{selected_row['batch_id']}_"
                                            )
                                            df['spectrum_column'] = unique_label + df[spectrum_id_col].astype(str)
                                            pivot_df = df.pivot_table(
                                                index='wavenumber',
                                                columns='spectrum_column',
                                                values='intensity'
                                            ).reset_index()
                                            pivot_df = pivot_df.rename(columns={'wavenumber': 'RamanShift'})
                                            df_this = pivot_df
                                            st.success(f"Fetched spectrum set for {class_label}.")
                                            st.dataframe(df_this.head())
                                            st.session_state[f'db_result_{idx}'] = df_this
                                        cur.close()
                                    except Exception as e:
                                        st.error(f"Database error: {e}")
                                        all_ready = False
                                    if f'db_result_{idx}' in st.session_state:
                                        df_this = st.session_state[f'db_result_{idx}']
                                    else:
                                        all_ready = False
                        else:
                            st.info("Select one result row for this class, then click Get Data.")
                            all_ready = False
                    else:
                        st.warning("No search results found.")
                        all_ready = False
                else:
                    st.info("Enter a keyword to search the database.")
                    all_ready = False

        class_labels.append(class_label)
        class_dfs.append((class_label, df_this if df_this is not None else None))

# --- Interpolate and combine all classes if ready ---
if all_ready and all(df is not None for _, df in class_dfs):
    st.success(f"All {n_classes} classes loaded successfully.")
    # For multi-class, interpolate/crop all to the common range
    mins, maxs = [], []
    for _, df in class_dfs:
        x = pd.to_numeric(df.iloc[:, 0])
        mins.append(x.min())
        maxs.append(x.max())
    common_min = int(np.ceil(max(mins)))
    common_max = int(np.floor(min(maxs)))
    if common_min >= common_max:
        st.error("Uploaded/selected spectra have no overlapping Raman‑shift range.")
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
    st.session_state.df = combined_df
    st.session_state.backup = combined_df

    # --- Label editor ---
    default_lbls_multi = {}
    for idx, (class_label, df_cls) in enumerate(interpolated_class_dfs, start=1):
        for col in df_cls.columns[1:]:
            default_lbls_multi[col] = idx
    show_label_editor(combined_df, default_lbls_multi)

elif all_ready and all(df is not None for _, df in class_dfs) and n_classes == 1:
    # If only one class, just show label editor for that class
    _, df = class_dfs[0]
    st.session_state.df = df
    st.session_state.backup = df
    st.session_state.class_data = class_dfs
    default_lbls_single = {col: 1 for col in df.columns[1:]}
    show_label_editor(df, default_lbls_single)

else:
    st.warning("Waiting for all classes to be ready. Make sure each class is fully uploaded or queried.")

# --- Downstream navigation/preview ---
if 'df' in st.session_state:
    st.divider()
    col1, col2 = st.columns([2, 12])
    if col1.button(label='Processing Page', key='switch_processing_page'):
        st.switch_page("pages/3_Processing.py")
    col2.markdown(' :arrow_left: **Go to Processing Page to process data**')
    st.write("#### Preview")
    st.write(st.session_state.backup)
    function.update_mode_option()
