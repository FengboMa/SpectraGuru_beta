# Introduction page
import streamlit as st
from streamlit_modal import Modal
# import streamlit.components.v1 as components

import function



function.wide_space_default()

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
# st.markdown("""
#         <style>
#                .block-container {
#                     padding-top: 0rem;
#                     padding-bottom: 0rem;
                    
#                 }
#         </style>
#         """, unsafe_allow_html=True)


# sidebar_icon = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v11\element\UGA_logo_ExtremeHoriz_FC_MARCM.png"
# st.logo(sidebar_icon, icon_image=sidebar_icon)

# st.image(r'C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v13\element\Phy2.png',
#             width = 1300)
st.image(r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\Application header picture-3.png")
st.session_state.log_file_path = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\user_count.txt"

hide_close_button_css = """
<style>
.modal-close {
    display: none !important;
}
</style>
"""

# Inject custom CSS to hide the close button
st.markdown(hide_close_button_css, unsafe_allow_html=True)
modal = Modal(
    "Welcome to SpectraGuru", 
    key="welcome_modal",
    
    # Optional
    padding=30, 
    max_width=750 
)

st.html(
    '''
        <style>
            div[aria-label="Modal"]>button[aria-label="Close"] {
                display: none;
            }
        </style>
    '''
)

if 'popup_closed' not in st.session_state:
    st.session_state.popup_closed = False

if not st.session_state.popup_closed:
    with modal.container():
        st.info('SpectraGuru is still under development. Current version: SpectraGuru ver. 0.14.2')
        
        st.write("")
        st.write("Thanks for visiting SpectraGuru, a spectroscopy processing and visualization tool.")
        st.write("If you encounter a problem, please send an email to Fengbo.Ma@uga.edu")
        st.divider()
        st.write("**:arrow_upper_left: After clicking Start button below, then navigate to Data Upload located at the top of the sidebar to begin!**")
        
        # check = st.button('Start', type='primary')
        # if check:
        #     st.session_state.popup_closed = True
        
        value = st.checkbox("By checking this box, you agree with data usage polices of SpectraGuru")
        if value:
            st.button('Start')
            st.session_state.popup_closed = True
            st.session_state.current_user_count = function.log_user_count(st.session_state.log_file_path)
        st.caption("More information Data Usage Policies, Disclaimers, License (link on hold)")

# -------------


st.write("# SpectraGuru  - A Spectra Analysis Application ")
st.info('SpectraGuru is still under development. Current version: SpectraGuru ver. 0.14.2')

# current_user_count = function.log_user_count(st.session_state.log_file_path)
try:
    counts = function.read_counts(st.session_state.log_file_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Views", st.session_state.current_user_count, None)
    col2.metric("Plots generated", counts['Plot_Generated'], None)
    col3.metric("Spectra processed", counts['Spectra_Processed'], None)
except:
    pass

st.sidebar.success("Navigate to Data Upload page above to start")

st.markdown(
    """
    SpectraGuru is a spectra analysis application designed to provide user-friendly tools for processing and visualizing spectra, aimed at accelerating your research. It functions as a dashboard or a specialized tool within a Python environment, organized with various modular functions that allow users to process spectroscopy data in a pipeline.
    
    ---
    
    """)

col1, col2 = st.columns([2, 12])
if col1.button(label='Data Upload Page', key='switch_data_upload_page'):
    st.switch_page("pages/Data_Upload.py")
col2.markdown(' :arrow_left: **Start with upload your data in Data Upload Page**')

st.divider()

col1, col3 = st.columns([2,3])

col1.markdown(
    """
    ### Features Include

    #### Data Upload Page
    **Support for specific format** 
    -   Data upload *(The file must follow certain formatting at this current version.)*
""")


col1.markdown(
    """
    #### Processing Page
    """)



col1.markdown(
    """
    **Processing**
    - Interpolation
    - Crop
    - Despike
    - Smoothening
    - Baseline removal
    - Normalization
    - Outlier removal
    """)
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")
# col2.write("")

col1.markdown(
    """
    **Visualization**
    - Preview Data
    - Interactive plotting
    - Fast Mode plotting
    - Export data
    """)

col3.image(r'C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\SpectraGuru Welcome Page Flow Chart.png')

# st.markdown(
#     """
    
#     """)
# col1, col2,col3 = st.columns([1,1,3])
col1.markdown(
    """
    #### Statistics Page
    - Average Plot with Original Spectra
    - Confidence Interval Plot
    - Correlation Heatmap
    """)

st.markdown(
    """
    ---

    ### About Us

    - Find us here: [Zhao Nano Lab](https://www.zhao-nano-lab.com/)
    - Explore more or report an issue? Send us a message to Fengbo (Fengbo.Ma@uga.edu)
"""
)

