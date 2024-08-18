# upload page
import streamlit as st
import pandas as pd
import function

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

loaded = False

if upload_file_format == None:
    if 'df' not in st.session_state:
        st.error("Please select a data format and upload your data")

elif upload_file_format == "Multi files: .txt files (two-column single spectrum files, common x)":
    uploaded_multi_file = st.file_uploader("", type=["txt"], key='multi_file_uploader', accept_multiple_files=True)
    # file_loaded = True
    try:
        if uploaded_multi_file is not None:
            loaded = True
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
        
if 'df' in st.session_state:
    st.divider()
    col1, col2 = st.columns([2, 12])
    
    if col1.button(label='Processing Page', key='switch_processing_page'):
        st.switch_page("pages/Processing.py")
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