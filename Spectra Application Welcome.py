from dotenv import load_dotenv
load_dotenv("CLERK.env")
import streamlit as st
from streamlit_modal import Modal
from auth_utils import clerk_signin_url, verify_clerk_session, request_proxy
# import streamlit.components.v1 as components

import function
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory
os.chdir(current_dir)


function.wide_space_default()

#####################
#params = st.query_params
#token  = params.get("__clerk_db_jwt") or params.get("session_id")
#if isinstance(token, list):
#    token = token[0]
#
#if token and 'user_logged_in' not in st.session_state:
#    user = verify_clerk_session(token)
#    if user:
#        st.session_state.user_logged_in = True
#        st.session_state.username    = user["first_name"]
#        st.session_state.popup_closed = True
#        st.query_params.clear() 
#        # Introduction page
#
#if 'user_logged_in' not in st.session_state:
#    st.session_state.user_logged_in = False
#
#print("DEBUG  token =", token)            # ← should be a long JWT string
#print("DEBUG  already_logged =", st.session_state.user_logged_in)

#####################


st.image(r"element/Application header picture-3.png")
st.session_state.log_file_path = r"element/user_count.txt"

# --- Initial Setup ---
hide_close_button_css = """
    <style>
        div[aria-label="Modal"]>button[aria-label="Close"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_close_button_css, unsafe_allow_html=True)


#request_proxy()

# ---------- grab token from URL on every run ----------



params = st.query_params                 # returns Mapping[str, str | list[str] | None]
token  = (
    params.get("__session")
    or params.get("__clerk_session")
    or params.get("__clerk_db_jwt")         # either str or list → handle both
    or params.get("session_id")
)
if isinstance(token, list):              # Clerk might give list
    token = token[0]
print("TOKEN:",token)

# ---------- session flags ----------
for key, default in {
    "popup_closed": False,
    "user_logged_in": False,
}.items():
    st.session_state.setdefault(key, default)

# ---------- get user info from token ----------
if False and token and not st.session_state.user_logged_in:
    user = verify_clerk_session(token)
    print("USER:",user)
    if user:
        st.session_state.user_logged_in = True
        st.session_state.username = user.get("first_name", "User")
        st.session_state.popup_closed = True        # skip modal from now on
        print("DEBUG: Retrieved user from token.")
    #st.query_params.clear()

# ---------- helper for guest button ----------
def guest_entry():
    st.session_state.user_logged_in = False
    st.session_state.popup_closed = True
    st.session_state.current_user_count = function.log_user_count(
        st.session_state.log_file_path
    )

# ---------- modal ----------
if not st.session_state.popup_closed and not st.session_state.user_logged_in:
    # hide close icon
    st.markdown(
        """
        <style>
        div[aria-label="Modal"]>button[aria-label="Close"] {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    modal = Modal("Welcome to SpectraGuru", key="welcome_modal",
                  padding=20, max_width=600)

    with modal.container():
        st.info("SpectraGuru is still under development. Current version: SpectraGuru ver. 1.2.1")
        st.write("Thanks for visiting SpectraGuru, a spectroscopy processing and visualization tool.")
        st.write("If you encounter a problem, please email Fengbo.Ma@uga.edu")
        st.write("**:arrow_upper_left: After starting, go to ‘Data Upload’ in the sidebar to begin!**")

        col1, col2 = st.columns(2)

        # left: guest
        col1.button("Continue as Guest", on_click=guest_entry)

        # right: login via Clerk—just a link
        signin_url = clerk_signin_url()           # already returns the full redirect URL
        col2.link_button("Log in", signin_url,type="primary")  


        st.caption(
            "By clicking any button you agree with the "
            "[Policy and Disclaimer of SpectraGuru]"
            "(https://fengboma.github.io/docs.spectraguru/docs/License-Policies-Disclaimers.html)"
        )

# ---------- greet authenticated users ----------
if st.session_state.get("user_logged_in"):
    st.write(f"Welcome {st.session_state.get('username', 'Guest')} 👋")
# -------------


st.write("# SpectraGuru  - A Spectra Analysis Application ")
# st.info('SpectraGuru is still under development. Current version: SpectraGuru ver. 0.15')

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
    
    **Visit our [documentation](https://fengboma.github.io/docs.spectraguru/) for more information about SpectraGuru.**
    
    ---
    
    """)

st.info("Check out our latest news and updates on SpectraGuru!")

# Add expander to show the flyer
with st.expander("View SpectraGuru Flyer (v2, Oct 26)"):
    st.image("news/Spectraguru flyer v2 oct26 (1).png", caption="SpectraGuru Flyer v2 – October 26", use_container_width=True)
st.divider()


col1, col2 = st.columns([2, 12])
if col1.button(label='Data Upload Page', key='switch_data_upload_page'):
    st.switch_page("pages/2_Data_Upload.py")
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
        - Auto despike method
        - Manual despike method
    - Smoothening
        - Savitzky-Golay filter
        - 1D Fast Fourier Transform filter
    - Baseline removal
        - AirPLS
        - ModPoly
        - Gaussian-Lorentzian Fitting
    - Normalization
        - Normalize by area
        - Normalize by Peak
        - Min-Max Normalization
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

col3.image(r'element/SpectraGuru Welcome Page Flow Chart.png')

# st.markdown(
#     """
    
#     """)
# col1, col2,col3 = st.columns([1,1,3])
col1.markdown(
    """
    #### Analytics Page
    - Average Plot with Original Spectra
    - Confidence Interval Plot
    - Spectra Derivation
    - Correlation Heatmap
    - Peak Identification and Stats
    - Hierarchically-clustered Heatmap
    - Principal Components Analysis (PCA)
    - T‑Distributed Stochastic Neighbor Embedding (t‑SNE)
    """)


st.markdown(
    """
    ---

    ### Visitor geographic map

    The geographic map visualizes where visiter from based on their latitude and longitude, with color intensity representing the frequency at each location. Each point on the map is derived from our record, showing the geographical distribution of occurrences.
"""
)

import streamlit.components.v1 as components
p = open(r"element/traffic_heatmap.html")
components.html(p.read(), scrolling=True, height=550)


st.markdown(
    """
    ---

    ### About Us 

    - Find us here: [Zhao Nano Lab](https://www.zhao-nano-lab.com/)
    - Explore more or report an issue? Send us a message to us (zhao-nano-lab@uga.edu)
"""
)

st.markdown(
    """
    ### SpectraGuru supported by:
    <div style="display:flex; justify-content:center; align-items:center; gap:40px;">
        <img src="element/USDA.png" style="height:100px; object-fit:contain;">
        <img src="element/nsf.png" style="height:100px; object-fit:contain;">
    </div>
    """,
    unsafe_allow_html=True
)