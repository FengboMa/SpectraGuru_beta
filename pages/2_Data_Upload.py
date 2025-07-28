# upload page
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
# sidebar_icon = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v11\element\UGA_logo_ExtremeHoriz_FC_MARCM.png"
# st.logo(sidebar_icon, icon_image=sidebar_icon)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8')
    return df

@st.cache_data
def load_tab_data(file):
    df = pd.read_csv(file, encoding='utf-8',sep='\t')
    return df

@st.cache_data
def load_multi_data(file_paths):
    def read_columns(file_path):
        try:            
            df = pd.read_csv(file_path, delim_whitespace=True, skip_blank_lines=True,
                            comment='#', header=None, on_bad_lines='skip')
            
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            # print(f"File: {file_path}, Shape: {df.shape}")
            if df.shape[1] < 2:
                raise ValueError(f"File {file_path} does not contain at least two columns.")
            return df[0], df[1]  # Return the first and second columns
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None
    
    first_columns = []
    second_columns = []
    file_names = []
    
    for file in file_paths:
        first_col, second_col = read_columns(file)
        if first_col is None or second_col is None:
            raise ValueError(f"Failed to read columns from file: {file}")
        first_columns.append(first_col)
        second_columns.append(second_col)
        file_names.append(file.name)

    if not first_columns:
        raise ValueError("No columns were read from the files.")

    # Check if all first columns are the same
    reference_column = first_columns[0]
    for column in first_columns[1:]:
        if not reference_column.equals(column):
            raise ValueError("The first column is not the same in all files.")
    
    # Append the second columns together
    df = pd.concat(second_columns, axis=1)
    df.insert(0, 'First_Column', reference_column)
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



""""""
st.write("## Data Upload")

# st.write("### Setting and upload")

option_map = {
    0: "Manually Upload",
    1: "Query from Database"
}

# Create the segmented control widget
selection = st.segmented_control(
    "Data Source",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
    default=0,
    help="Select how you want to upload your data."
)

# Display the selected option
st.write(
    "Your selected option:",
    f"{None if selection is None else option_map[selection]}"
)

# Conditional logic based on the selected option
if selection == 0:

    # st.markdown("""
    #             *Upload your spectra data here*
                
    #             Notice that: The file must follow certain formatting at this current version.
                
    #             """)

    # upload_file_format = st.selectbox(label="Select data format you wish to upload", 
    #                                 options=("Multi files: .txt files (two-column single spectrum files, common x)",
    #                                         "Multi files: .csv files (two-column single spectrum files, common x)",
    #                                         "Single file: .csv file (A tab-separated csv (tsv) file)",
    #                                         "Single file: .csv file (A comma-separated csv (csv) file)"),
    #                                 help="See documentation for detailed description on supported format.",
    #                                 placeholder="Please choose a format",
    #                                 index = None)

    # # if upload_file_format is None:
    # #     st.error("Please select a data format first!")
    # #     st.stop()

    # multi_class = st.checkbox("Upload multiple classes?", value=False)

    # if multi_class:
    #     # Ask how many classes (2 to 10)
    #     n_classes = st.number_input(
    #         "How many classes?",
    #         min_value=2,
    #         max_value=10,
    #         value=2,
    #         step=1
    #     )
        
    #     # We'll collect each class’s upload into a list
    #     class_uploads = []
        
    #     for i in range(int(n_classes)):
    #         label = f"Class #{i+1} upload"
            
    #         # Mirror the format choice when building each uploader:
    #         if upload_file_format.startswith("Multi files: .txt"):
    #             files = st.file_uploader(
    #                 label, type=["txt"],
    #                 accept_multiple_files=True,
    #                 key=f"class_{i}"
    #             )
    #             class_uploads.append(("txt", files))
            
    #         elif upload_file_format.startswith("Multi files: .csv"):
    #             files = st.file_uploader(
    #                 label, type=["csv"],
    #                 accept_multiple_files=True,
    #                 key=f"class_{i}"
    #             )
    #             class_uploads.append(("csv_multi", files))
            
    #         elif "tab-separated" in upload_file_format:
    #             f = st.file_uploader(
    #                 label, type=["csv"],  # TSVs are still .csv
    #                 key=f"class_{i}"
    #             )
    #             class_uploads.append(("tsv", f))
            
    #         elif "comma-separated" in upload_file_format:
    #             f = st.file_uploader(
    #                 label, type=["csv"],
    #                 key=f"class_{i}"
    #             )
    #             class_uploads.append(("csv", f))
        
    #     # Only once *all* classes have files uploaded do we process:
    #     if all(upload for fmt, upload in class_uploads):
    #         data_per_class = []
    #         for fmt, upload in class_uploads:
    #             if fmt == "txt":
    #                 # multi‑txt loader expects a list of files
    #                 df = load_multi_data(upload)
    #             elif fmt == "csv_multi":
    #                 df = load_multi_data(upload)
    #             elif fmt == "tsv":
    #                 df = load_tab_data(upload)
    #             else:  # "csv"
    #                 df = load_data(upload)
                
    #             data_per_class.append(df)
            
    #         # e.g. stash in session_state or move on to analysis
    #         st.session_state.class_data = data_per_class
    #         st.success(f"Loaded {len(data_per_class)} classes.")

    # else:

    #     loaded = False

    #     if upload_file_format == None:
    #         if 'df' not in st.session_state:
    #             st.error("Please select a data format and upload your data")
                
    #             #Sample data
    #             st.write("")
    #             st.write('Don\'t have data? Try with sample data.' )
                
    #             if st.button(label='Load sample data'):
    #                 df = load_data(r'element/sampledata.csv')
                    
    #                 st.session_state.df = df
    #                 st.session_state.backup = df


    #     elif upload_file_format == "Multi files: .txt files (two-column single spectrum files, common x)":
    #         uploaded_multi_file = st.file_uploader("", type=["txt"], key='multi_file_uploader', accept_multiple_files=True)
    #         # file_loaded = True
    #         try:
    #             if uploaded_multi_file is not None:
    #                 # loaded = True
    #                 # df = load_multi_data(uploaded_multi_file)
                    
    #                 # st.session_state.df = df
    #                 # st.session_state.backup = df
    #                 loaded = True
    #                 # df = load_multi_data(uploaded_multi_file)
                    
    #                 # st.session_state.df = load_multi_data(uploaded_multi_file)
    #                 # st.session_state.backup = load_multi_data(uploaded_multi_file)
                    
    #                 df = load_multi_data(uploaded_multi_file)
                    
    #                 st.session_state.df = df
    #                 st.session_state.backup = df
                    
    #         except:
    #             pass
    #     elif upload_file_format == "Multi files: .csv files (two-column single spectrum files, common x)":
    #         uploaded_multi_file = st.file_uploader("", type=["csv"], key='multi_file_uploader', accept_multiple_files=True)
    #         # file_loaded = True
    #         try:
    #             if uploaded_multi_file is not None:
    #                 loaded = True
    #                 df = load_multi_data(uploaded_multi_file)
                    
    #                 st.session_state.df = df
    #                 st.session_state.backup = df
    #         except:
    #             pass
    #     elif upload_file_format == "Single file: .csv file (A tab-separated csv (tsv) file)":
    #         uploaded_file = st.file_uploader("", type="csv", key='file_uploader')
    #         # file_loaded = True
    #         try: 
    #             if uploaded_file is not None:
    #                 loaded = True
    #                 df = load_tab_data(uploaded_file)

    #                 st.session_state.df = df
    #                 st.session_state.backup = df
    #         except:
    #             pass
    #     elif upload_file_format == "Single file: .csv file (A comma-separated csv (csv) file)":
    #         uploaded_file = st.file_uploader("", type="csv", key='file_uploader')
    #         # file_loaded = True
    #         try: 
    #             if uploaded_file is not None:
    #                 loaded = True
    #                 df = load_data(uploaded_file)

    #                 st.session_state.df = df
    #                 st.session_state.backup = df
    #         except:
    #             pass



    # ------------------ CONFIG ------------------
    FORMAT_OPTIONS = {
        "Multi TXT (two-column, common x)": {"kind": "multi_txt", "multi": True,  "types": ["txt"]},
        "Multi CSV (two-column, common x)": {"kind": "multi_csv", "multi": True,  "types": ["csv"]},
        "Single TSV (.tsv / tab-separated)": {"kind": "single_tsv", "multi": False, "types": ["csv"]},
        "Single CSV (comma-separated)":      {"kind": "single_csv", "multi": False, "types": ["csv"]},
    }

    st.write("Class‑Aware Data Upload")

    # ----------------------  NUMBER OF CLASSES  --------------------
    n_classes = st.number_input(
        "Number of classes",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help=(
            "• **1** → Single‑class upload (all spectra belong to one class).  \n"
            "• **2 – 10** → Multi‑class upload. You will be asked to select a "
            "format and upload files for each class separately."
        )
    )

    use_labels = False  # flip to True if you re‑enable custom labels

    # --------------------  HELPER DISPATCHER  ----------------------
    def process_upload(kind, uploaded):
        if kind in ("multi_txt", "multi_csv"):
            return load_multi_data(uploaded) if uploaded else None
        if kind == "single_tsv":
            return load_tab_data(uploaded)   if uploaded is not None else None
        if kind == "single_csv":
            return load_data(uploaded)       if uploaded is not None else None
        return None


    # =================================================================
    #                       SINGLE‑CLASS MODE
    # =================================================================
    if n_classes == 1:
        st.subheader("Single‑Class Upload")

        single_format = st.selectbox(
            "Select file format",
            list(FORMAT_OPTIONS.keys()),
            index=None,
            placeholder="Choose a format ..."
        )

        label_input = "Class_1" if not use_labels else st.text_input("Class label", value="Class_1")

        df_loaded, uploaded_any = None, False

        if single_format:
            cfg = FORMAT_OPTIONS[single_format]
            uploader_label = "Upload file(s)" if cfg["multi"] else "Upload file"
            if cfg["multi"]:
                files = st.file_uploader(
                    uploader_label, type=cfg["types"],
                    accept_multiple_files=True, key="single_multi_uploader"
                )
                uploaded_any = bool(files)
                df_loaded = process_upload(cfg["kind"], files)
            else:
                file_obj = st.file_uploader(
                    uploader_label, type=cfg["types"], key="single_file_uploader"
                )
                uploaded_any = file_obj is not None
                df_loaded = process_upload(cfg["kind"], file_obj)

        if df_loaded is not None:
            st.success(f"Loaded dataset ({df_loaded.shape[0]} rows × {df_loaded.shape[1]} cols).")
            st.dataframe(df_loaded.head())
            st.session_state.df = df_loaded
            st.session_state.backup = df_loaded
            st.session_state.class_data = [(label_input, df_loaded)]
            # ---------- Build & show label editor ----------
            default_lbls = {col: 1 for col in df_loaded.columns[1:]}   # skip RamanShift
            show_label_editor(df_loaded, default_lbls)

        else:
            st.info("Awaiting upload..." if single_format else "Select a format to begin.")
            if 'df' not in st.session_state and not uploaded_any:
                st.error("No data uploaded.")
                st.write("Don't have data? Try with sample data.")
                if st.button("Load sample data"):
                    try:
                        df_sample = load_data(r'element/sampledata.csv')
                        st.session_state.df = df_sample
                        st.session_state.backup = df_sample
                        st.session_state.class_data = [("Sample", df_sample)]
                        st.success("Sample data loaded.")
                        # st.dataframe(df_sample.head())
                    except Exception as e:
                        st.error(f"Failed to load sample data: {e}")


    # =================================================================
    #                      MULTI‑CLASS MODE
    # =================================================================
    else:
        st.subheader("Multi‑Class Uploads")
        st.warning(
        "All uploaded classes will be **automatically interpolated** to a "
        "common 1 cm⁻¹ Raman‑shift grid and then **cropped** to the shared "
        "overlap range before further analysis. "
        "This ensures every spectrum has identical x‑axis values."
        )
        class_dfs = []
        all_ready = True

        for idx in range(int(n_classes)):
            with st.expander(f"Class {idx + 1}", expanded=True):

                class_label = (f"Class_{idx + 1}" if not use_labels else
                            st.text_input(f"Label for Class {idx + 1}",
                                            value=f"Class_{idx + 1}",
                                            key=f"class_label_{idx}"))

                fmt = st.selectbox(
                    f"Format for {class_label}",
                    list(FORMAT_OPTIONS.keys()),
                    index=None,
                    placeholder="Choose a format ...",
                    key=f"class_fmt_{idx}"
                )

                df_this = None
                if fmt:
                    cfg = FORMAT_OPTIONS[fmt]
                    if cfg["multi"]:
                        uploads = st.file_uploader(
                            f"Upload file(s) for {class_label}",
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
                            f"Upload file for {class_label}",
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

                # ---------- Preview directly inside this expander ----------
                if df_this is not None:
                    st.success(f"{class_label} loaded. Shape: {df_this.shape}")
                    st.dataframe(df_this.head())
                else:
                    st.info("Waiting for format & upload." if fmt else "Select a format.")

                class_dfs.append((class_label, df_this))

        # Final readiness check
        if all_ready and all(df is not None for _, df in class_dfs):
            st.success(f"All {n_classes} classes loaded successfully.")

            # ----------------- 1. Determine common crop range -----------------
            # Each df's first column must be numeric Raman shift values
            mins = []
            maxs = []
            for _, df in class_dfs:
                x = pd.to_numeric(df.iloc[:, 0])
                mins.append(x.min())
                maxs.append(x.max())
            common_min = int(np.ceil(max(mins)))   # upper of minima
            common_max = int(np.floor(min(maxs)))  # lower of maxima

            if common_min >= common_max:
                st.error("Uploaded spectra have no overlapping Raman‑shift range.")
                st.stop()

            # Integer‑spaced reference axis
            ref_x = np.arange(common_min, common_max + 1, 1)

            # Store for later pages if you already use that name elsewhere
            st.session_state.interpolation_ref_x = ref_x

            # ----------------- 2. Interpolate & crop per class ----------------
            interpolated_class_dfs = []
            for class_label, df in class_dfs:
                x_orig = pd.to_numeric(df.iloc[:, 0])
                interp_df = pd.DataFrame({'RamanShift': ref_x})

                # iterate over each spectrum column
                for col in df.columns[1:]:
                    f = interp1d(
                        x_orig,
                        pd.to_numeric(df[col], errors='coerce'),
                        kind='linear',
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    interp_df[col] = f(ref_x)

                # Ensure unique column names: prepend class label
                rename_map = {c: f"{class_label}__{c}" for c in interp_df.columns[1:]}
                interp_df.rename(columns=rename_map, inplace=True)

                interpolated_class_dfs.append((class_label, interp_df))

            # ----------------- 3. Combine all classes -------------------------
            combined_df = (
                pd.concat([df.set_index('RamanShift') for _, df in interpolated_class_dfs],
                        axis=1)
                .reset_index()
                .drop_duplicates()
            )

            # ----------------- 4. Persist in session_state -------------------
            st.session_state.class_data = interpolated_class_dfs   # list[(label, df)]
            st.session_state.df = combined_df
            st.session_state.backup = combined_df
                # ---------- Build & show label editor ----------
            # Assign default label = class index (1‑based)
            default_lbls_multi = {}
            for idx, (class_label, df_cls) in enumerate(interpolated_class_dfs, start=1):
                for col in df_cls.columns[1:]:               # skip RamanShift
                    default_lbls_multi[col] = idx
            show_label_editor(combined_df, default_lbls_multi)


            # # ----------------- 5. Show summary -------------------------------
            # st.markdown("### Summary (after interpolation & crop)")
            # st.write(f"Common range: **{common_min} – {common_max} cm⁻¹** "
            #         f"({len(ref_x)} points)")
            # st.dataframe(combined_df.iloc[:5, :10])  # preview first rows/cols

        else:
            st.warning("Waiting for all class formats and uploads before processing.")

elif selection == 1:
    # st.session_state.connection = None

    with st.form("DB_login"):
        st.write("Log in to DB")
        st.session_state.user = st.text_input("User")
        st.session_state.passkey = st.text_input("Passkey")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Log in")
        if submitted:
            try:
                conn = psycopg2.connect(
                    dbname="SpectraGuruDB",
                    user=st.session_state.user,
                    password=st.session_state.passkey,
                    host="localhost",  # Use "localhost" for local database
                    port="5432"  # Default PostgreSQL port
                )
                cur = conn.cursor()
                
                st.info("Connection to the database was successful.")
                st.session_state.connection = True

            except:
                # If an error occurs, print the error
                st.warning(f"The error occurred")



    
    # st.write(st.session_state.connection)
    try:
        if st.session_state.connection:
            st.write("## Database Search")

            advanced_search = st.checkbox("Advanced Search")

            # If advanced search is selected, show additional filters
            if advanced_search:
                data_type_filter = st.radio(
                    "Select Data Type to Search:",
                    options=["Both", "Raw Data Only", "Standard Data Only"],
                    index=0,
                    horizontal=True
                )
            else:
                data_type_filter = "Both"

            search_query = st.text_input("Enter a keyword to search in the database:", "")
            st.write("### Search Results")


            if search_query:
                results = function.search_database(search_query, data_type_filter)
                
                if not results.empty:
                    st.write("### Search Results")
                    
                    # Step 1: Add 'Select' column (default to False)
                    results['Select'] = False

                    # Step 2: Reorder columns
                    columns_order = ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type'] + \
                                    [col for col in results.columns if col not in ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type']]
                    results = results[columns_order]

                    # Step 3: Display editable dataframe with checkbox
                    edited_results = st.data_editor(results, column_config={"Select": st.column_config.CheckboxColumn()})
                    
                    

                    # Step 4: Filter selected rows and print
                    selected_rows = edited_results[edited_results['Select'] == True]
                    if not selected_rows.empty:
                        selected_strings = [
                            f"{row['batch_analyte_name']}_{row['batch_data_type']}_{row['batch_id']}"
                            for _, row in selected_rows.iterrows()
                        ]
                        selection_output = "; ".join(selected_strings)
                        st.write(f"You selected: {selection_output}")
                    else:
                        st.write("No rows selected.")
                    
                    # if st.button("Get Data"):
                    if len(selected_rows) == 1:
                        selected_row = selected_rows.iloc[0]
                        batch_id = selected_row['batch_id']
                        batch_data_type = selected_row['batch_data_type']  # Assume you store this info

                        # Connect to the database
                        conn = function.get_db_connection()
                        cur = conn.cursor()

                        if batch_data_type.lower() == 'raw':
                            query = """
                                SELECT sd.*
                                FROM spectrum s
                                JOIN spectrum_data sd ON s.spectrum_id = sd.spectrum_id
                                WHERE s.batch_id = %s
                            """
                            # st.write(batch_data_type)
                            # st.write('xxxx')
                        elif batch_data_type.lower() == 'standard':
                            query = """
                                SELECT sds.*
                                FROM spectrum_standard ss
                                JOIN spectrum_data_standard sds ON ss.spectrum_standard_id = sds.spectrum_standard_id
                                WHERE ss.batch_standard_id = %s
                            """
                            # st.write(batch_data_type)
                            # st.write('yyyy')
                        else:
                            st.warning("Unknown batch data type selected.")
                            query = None
                        if st.button("Get Data"):
                            try:
                                if query:
                                    with st.status("Fetching spectrum data from SpectraGuru Database...", expanded=True) as status:
                                        cur.execute(query, (int(batch_id),))
                                        spectrum_data = cur.fetchall()
                                        column_names = [desc[0] for desc in cur.description]
                                        spectrum_df = pd.DataFrame(spectrum_data, columns=column_names)
                                        # st.write('zzzz')
                                    if not spectrum_df.empty:
                                        # st.write('111')
                                        # st.write("### Spectrum Data")
                                        # st.dataframe(spectrum_df)
                                        # st.divider()
                                        spectrum_id_col = spectrum_df.columns[1]  # Second column

                                        # Step 2: Check required columns
                                        required_cols = ['wavenumber', 'intensity']
                                        missing_cols = [col for col in required_cols if col not in spectrum_df.columns]
                                        if missing_cols:
                                            raise ValueError(f"Missing columns in spectrum_df: {missing_cols}")

                                        # Step 3: Build the unique column name from selected_row
                                        unique_col_name = (
                                            f"{selected_row['batch_analyte_name']}_"
                                            f"{selected_row['batch_data_type']}_"
                                            f"{selected_row['batch_id']}_"
                                        )

                                        # Step 4: Append the spectrum ID from the dynamic column
                                        spectrum_df['spectrum_column'] = unique_col_name + spectrum_df[spectrum_id_col].astype(str)

                                        # Step 5: Pivot
                                        pivot_df = spectrum_df.pivot_table(
                                            index='wavenumber',
                                            columns='spectrum_column',
                                            values='intensity'
                                        ).reset_index()

                                        # Step 6: Rename 'wavenumber' to 'RamanShift'
                                        pivot_df = pivot_df.rename(columns={'wavenumber': 'RamanShift'})


                                        # st.write("### Pivoted Spectrum Data")
                                        # st.dataframe(pivot_df)
                                        st.session_state.df = pivot_df
                                        st.session_state.backup = pivot_df
                                else:
                                        st.warning("No spectrum data found for this batch.")
                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                            finally:
                                cur.close()
                                conn.close()
                    else:
                        st.info("Please select exactly one row to view spectrum data.")
                        
                else:
                    st.warning("No results found.")

                    st.divider()
    # except:
    #     st.warning("Please log in to the database first.")
    except Exception as e:
        print("An error occurred:", e)


############

if 'df' in st.session_state:
    st.divider()
    col1, col2 = st.columns([2, 12])
    
    if col1.button(label='Processing Page', key='switch_processing_page'):
        st.switch_page("pages/3_Processing.py")
    col2.markdown(' :arrow_left: **Go to Processing Page to process data**')
    
    st.write("#### Preview")
    st.write(st.session_state.backup)
    
    function.update_mode_option()
# elif upload_file_format is not None or 'backup' not in st.session_state or load_error == True:
# elif 'df' not in st.session_state and load_error == True:
# st.write(loaded)
# st.write('df' in st.session_state)
# st.write(upload_file_format is None)
#         st.error("Data upload is failed. Please check your data format.")