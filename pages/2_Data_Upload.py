# upload page
import streamlit as st
import pandas as pd
import function
import psycopg2

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

    st.markdown("""
                *Upload your spectra data here*
                
                Notice that: The file must follow certain formatting at this current version.
                
                """)

    upload_file_format = st.selectbox(label="Select data format you wish to upload", 
                                    options=("Multi files: .txt files (two-column single spectrum files, common x)",
                                            "Multi files: .csv files (two-column single spectrum files, common x)",
                                            "Single file: .csv file (A tab-separated csv (tsv) file)",
                                            "Single file: .csv file (A comma-separated csv (csv) file)"),
                                    help="See documentation for detailed description on supported format.",
                                    placeholder="Please choose a format",
                                    index = None)

    # if upload_file_format is None:
    #     st.error("Please select a data format first!")
    #     st.stop()

    multi_class = st.checkbox("Upload multiple classes?", value=False)

    if multi_class:
        # Ask how many classes (2 to 10)
        n_classes = st.number_input(
            "How many classes?",
            min_value=2,
            max_value=10,
            value=2,
            step=1
        )
        
        # We'll collect each class’s upload into a list
        class_uploads = []
        
        for i in range(int(n_classes)):
            label = f"Class #{i+1} upload"
            
            # Mirror the format choice when building each uploader:
            if upload_file_format.startswith("Multi files: .txt"):
                files = st.file_uploader(
                    label, type=["txt"],
                    accept_multiple_files=True,
                    key=f"class_{i}"
                )
                class_uploads.append(("txt", files))
            
            elif upload_file_format.startswith("Multi files: .csv"):
                files = st.file_uploader(
                    label, type=["csv"],
                    accept_multiple_files=True,
                    key=f"class_{i}"
                )
                class_uploads.append(("csv_multi", files))
            
            elif "tab-separated" in upload_file_format:
                f = st.file_uploader(
                    label, type=["csv"],  # TSVs are still .csv
                    key=f"class_{i}"
                )
                class_uploads.append(("tsv", f))
            
            elif "comma-separated" in upload_file_format:
                f = st.file_uploader(
                    label, type=["csv"],
                    key=f"class_{i}"
                )
                class_uploads.append(("csv", f))
        
        # Only once *all* classes have files uploaded do we process:
        if all(upload for fmt, upload in class_uploads):
            data_per_class = []
            for fmt, upload in class_uploads:
                if fmt == "txt":
                    # multi‑txt loader expects a list of files
                    df = load_multi_data(upload)
                elif fmt == "csv_multi":
                    df = load_multi_data(upload)
                elif fmt == "tsv":
                    df = load_tab_data(upload)
                else:  # "csv"
                    df = load_data(upload)
                
                data_per_class.append(df)
            
            # e.g. stash in session_state or move on to analysis
            st.session_state.class_data = data_per_class
            st.success(f"Loaded {len(data_per_class)} classes.")

    else:

        loaded = False

        if upload_file_format == None:
            if 'df' not in st.session_state:
                st.error("Please select a data format and upload your data")
                
                #Sample data
                st.write("")
                st.write('Don\'t have data? Try with sample data.' )
                
                if st.button(label='Load sample data'):
                    df = load_data(r'element/sampledata.csv')
                    
                    st.session_state.df = df
                    st.session_state.backup = df


        elif upload_file_format == "Multi files: .txt files (two-column single spectrum files, common x)":
            uploaded_multi_file = st.file_uploader("", type=["txt"], key='multi_file_uploader', accept_multiple_files=True)
            # file_loaded = True
            try:
                if uploaded_multi_file is not None:
                    # loaded = True
                    # df = load_multi_data(uploaded_multi_file)
                    
                    # st.session_state.df = df
                    # st.session_state.backup = df
                    loaded = True
                    # df = load_multi_data(uploaded_multi_file)
                    
                    # st.session_state.df = load_multi_data(uploaded_multi_file)
                    # st.session_state.backup = load_multi_data(uploaded_multi_file)
                    
                    df = load_multi_data(uploaded_multi_file)
                    
                    st.session_state.df = df
                    st.session_state.backup = df
                    
            except:
                pass
        elif upload_file_format == "Multi files: .csv files (two-column single spectrum files, common x)":
            uploaded_multi_file = st.file_uploader("", type=["csv"], key='multi_file_uploader', accept_multiple_files=True)
            # file_loaded = True
            try:
                if uploaded_multi_file is not None:
                    loaded = True
                    df = load_multi_data(uploaded_multi_file)
                    
                    st.session_state.df = df
                    st.session_state.backup = df
            except:
                pass
        elif upload_file_format == "Single file: .csv file (A tab-separated csv (tsv) file)":
            uploaded_file = st.file_uploader("", type="csv", key='file_uploader')
            # file_loaded = True
            try: 
                if uploaded_file is not None:
                    loaded = True
                    df = load_tab_data(uploaded_file)

                    st.session_state.df = df
                    st.session_state.backup = df
            except:
                pass
        elif upload_file_format == "Single file: .csv file (A comma-separated csv (csv) file)":
            uploaded_file = st.file_uploader("", type="csv", key='file_uploader')
            # file_loaded = True
            try: 
                if uploaded_file is not None:
                    loaded = True
                    df = load_data(uploaded_file)

                    st.session_state.df = df
                    st.session_state.backup = df
            except:
                pass


    # multi_file = st.checkbox("Uploading Multiple Files", 
    #                     value=True, 
    #                     help = """
    #                     Check box if the files are sharing common X-axis and for the same sample. The csv or txt file should be separated by tab.
    #                     """)

    # if multi_file is True:
    #     st.write("#### Upload Multiple Files")
    #     # Multi file uploader
    #     uploaded_multi_file = st.file_uploader("", type=["csv","txt"], key='multi_file_uploader', accept_multiple_files=True)
    #     try:
    #         if uploaded_multi_file is not None:
    #             df = load_multi_data(uploaded_multi_file)
                
    #             st.session_state.df = df
    #             st.session_state.backup = df
    #         # if 'df' in st.session_state:
                
    #         #     st.write("#### Preview")
    #         #     st.write(st.session_state.df)
    #             # st.write(st.session_state.df_original)
    #     except:
    #         print("Load your data")

    # else: 
    #     st.write("#### Upload a Single File")
        
    #     st.write("Check the box if it is a tab-separated csv file")
        
    #     tab_csv = st.checkbox("This is a tab separated csv file", 
    #                         value=True, 
    #                         help = """
    #                         Check box if the file is a tab-separated csv (tsv).                                                        
    #                         A tab-separated csv (tsv) file is a type of text file used
    #                         to store data in a tabular format, similar to a standard csv file, but 
    #                         with tab characters ("/t") used as the delimiter instead of commas. This format 
    #                         is useful for ensuring that data containing commas remains correctly parsed.""")
        
    #     uploaded_file = st.file_uploader("", type="csv", key='file_uploader')

    #     if uploaded_file is not None:
            
    #         if tab_csv == False:
    #             df = load_data(uploaded_file)
    #         elif tab_csv == True:
    #             df = load_tab_data(uploaded_file)
            
    #         st.session_state.df = df
    #         st.session_state.backup = df
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

    def get_db_connection():
        return psycopg2.connect(
            dbname="SpectraGuruDB",
            user=st.session_state.user,
            password=st.session_state.passkey,
            host="localhost",
            port="5432"
        )

    def compute_relevance(row, keywords):
        return sum(
            any(str(keyword).lower() in str(cell).lower() for cell in row)
            for keyword in keywords
        )

    # Search Function
    def search_database(search_term, data_type_filter="Both"):
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            search_pattern = f"%{search_term}%"

            # Query 1 (Search in Raw Data)
            query1 = """
            SELECT 
                u.user_id AS user_id,
                u.name AS user_name,
                u.location AS user_location,
                u.institution AS user_institution,
                p.project_id AS project_id,
                p.project_name AS project_name,
                p.start_date AS project_start_date,
                p.source AS project_source,
                db.batch_id AS batch_id,  -- Keep consistent column names
                db.upload_date AS batch_upload_date,
                db.analyte_name AS batch_analyte_name,
                db.buffer_solution AS batch_buffer_solution,
                db.instrument_details AS batch_instrument_details,
                db.wavelength AS batch_wavelength,
                db.power AS batch_power,
                db.concentration AS batch_concentration,
                db.concentration_units AS batch_concentration_units,
                db.accumulation_time AS batch_accumulation_time,
                db.experimental_procedure AS batch_experimental_procedure,
                db.substrate_type AS batch_substrate_type,
                db.substrate_material AS batch_substrate_material,
                db.preparation_conditions AS batch_preparation_conditions,
                db.data_type AS batch_data_type,
                db.notes AS batch_notes
            FROM
                "user" u
            JOIN
                project_user pu ON u.user_id = pu.user_id
            JOIN
                "project" p ON p.project_id = pu.project_id
            JOIN
                project_batch pb ON p.project_id = pb.project_id
            JOIN
                "databatch" db ON db.batch_id = pb.batch_id
            WHERE 
                COALESCE(u.name, '') ILIKE %s OR 
                COALESCE(u.location, '') ILIKE %s OR 
                COALESCE(u.institution, '') ILIKE %s OR 
                COALESCE(p.project_name, '') ILIKE %s OR
                COALESCE(p.source, '') ILIKE %s OR 
                COALESCE(db.analyte_name, '') ILIKE %s OR
                COALESCE(db.buffer_solution, '') ILIKE %s OR
                COALESCE(db.instrument_details, '') ILIKE %s OR
                COALESCE(db.experimental_procedure, '') ILIKE %s OR
                COALESCE(db.substrate_type, '') ILIKE %s OR
                COALESCE(db.substrate_material, '') ILIKE %s OR
                COALESCE(db.preparation_conditions, '') ILIKE %s OR
                COALESCE(db.data_type, '') ILIKE %s OR
                COALESCE(db.notes, '') ILIKE %s;
            """

            # Query 2 (Search in Standard Data)
            query2 = """
            SELECT 
                u.user_id AS user_id,
                u.name AS user_name,
                u.location AS user_location,
                u.institution AS user_institution,
                p.project_id AS project_id,
                p.project_name AS project_name,
                p.start_date AS project_start_date,
                p.source AS project_source,
                db.batch_standard_id AS batch_id,  -- Keep column names same as Query 1
                db.upload_date AS batch_upload_date,
                db.analyte_name AS batch_analyte_name,
                db.buffer_solution AS batch_buffer_solution,
                db.instrument_details AS batch_instrument_details,
                db.wavelength AS batch_wavelength,
                db.power AS batch_power,
                db.concentration AS batch_concentration,
                db.concentration_units AS batch_concentration_units,
                db.accumulation_time AS batch_accumulation_time,
                db.experimental_procedure AS batch_experimental_procedure,
                db.substrate_type AS batch_substrate_type,
                db.substrate_material AS batch_substrate_material,
                db.preparation_conditions AS batch_preparation_conditions,
                db.data_type AS batch_data_type,
                db.notes AS batch_notes
            FROM
                "user" u
            JOIN
                project_user pu ON u.user_id = pu.user_id
            JOIN
                "project" p ON p.project_id = pu.project_id
            JOIN
                project_batch_standard pb ON p.project_id = pb.project_id
            JOIN
                "databatch_standard" db ON db.batch_standard_id = pb.batch_standard_id
            WHERE 
                COALESCE(u.name, '') ILIKE %s OR 
                COALESCE(u.location, '') ILIKE %s OR 
                COALESCE(u.institution, '') ILIKE %s OR 
                COALESCE(p.project_name, '') ILIKE %s OR
                COALESCE(p.source, '') ILIKE %s OR 
                COALESCE(db.analyte_name, '') ILIKE %s OR
                COALESCE(db.buffer_solution, '') ILIKE %s OR
                COALESCE(db.instrument_details, '') ILIKE %s OR
                COALESCE(db.experimental_procedure, '') ILIKE %s OR
                COALESCE(db.substrate_type, '') ILIKE %s OR
                COALESCE(db.substrate_material, '') ILIKE %s OR
                COALESCE(db.preparation_conditions, '') ILIKE %s OR
                COALESCE(db.data_type, '') ILIKE %s OR
                COALESCE(db.notes, '') ILIKE %s;
            """
            results = []

            keywords = search_term.strip().split()
            if not keywords:
                return pd.DataFrame()

            # Get full OR pattern
            like_patterns = [f"%{k}%" for k in keywords]

            # Set of results
            results = []

            for pattern in like_patterns:
                params = (pattern,) * 14

                if data_type_filter in ["Both", "Raw Data Only"]:
                    cur.execute(query1, params)
                    rows1 = cur.fetchall()
                    columns1 = [desc[0] for desc in cur.description]
                    df1 = pd.DataFrame(rows1, columns=columns1)
                    results.append(df1)

                if data_type_filter in ["Both", "Standard Data Only"]:
                    cur.execute(query2, params)
                    rows2 = cur.fetchall()
                    columns2 = [desc[0] for desc in cur.description]
                    df2 = pd.DataFrame(rows2, columns=columns2)
                    results.append(df2)

            if results:
                result_df = pd.concat(results, ignore_index=True).drop_duplicates()

                # Add a relevance score column
                result_df["relevance"] = result_df.apply(lambda row: compute_relevance(row, keywords), axis=1)

                # Sort by relevance descending
                result_df = result_df.sort_values(by="relevance", ascending=False)
            else:
                result_df = pd.DataFrame()

            
            return result_df
            # # Execute Query 1
            # cur.execute(query1, (search_pattern,) * 14)  # Pass correct number of parameters
            # rows1 = cur.fetchall()
            # columns1 = [desc[0] for desc in cur.description]
            # df1 = pd.DataFrame(rows1, columns=columns1)

            # # Execute Query 2
            # cur.execute(query2, (search_pattern,) * 14)  # Pass correct number of parameters
            # rows2 = cur.fetchall()
            # columns2 = [desc[0] for desc in cur.description]
            # df2 = pd.DataFrame(rows2, columns=columns2)

            # # Merge results into one DataFrame
            # result_df = pd.concat([df1, df2], ignore_index=True)

            # conn.close()
            # return result_df
        except Exception as e:
            conn.close()
            st.error(f"Error fetching search results: {e}")
            return pd.DataFrame()
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



            if search_query:
                results = search_database(search_query, data_type_filter)
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
                        conn = get_db_connection()
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