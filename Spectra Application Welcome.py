import streamlit as st
from streamlit_modal import Modal
from auth_utils import LOCAL_DEPLOY, login, logout, startup, populate
from auth_utils import clerk_component
# import streamlit.components.v1 as components

import function
import log_utils as log
import os
import base64

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

if 'user_decided' not in st.session_state:
    st.session_state.user_decided = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'do_startup' not in st.session_state:
    st.session_state.do_startup = True
if 'show_welcome_modal' not in st.session_state:
    st.session_state.show_welcome_modal = True
if 'show_login_modal' not in st.session_state:
    st.session_state.show_login_modal = False

if st.session_state.do_startup:
    print("DOING STARTUP...")
    startup()
else:
    print("STARTUP SKIPPED")

#st.write(st.session_state.user)
print("User:", st.session_state.user)

#log.clear_call_log()
#log.clear_count_log()

st.image(r"element/Application header picture-3.png")
#st.session_state.log_file_path = r"element/user_count.txt"

# Initialize user count (logs +1 for each unique session)
if 'current_user_count' not in st.session_state:
    #st.session_state.current_user_count = function.log_user_count(
    #    st.session_state.log_file_path
    #)
    st.session_state.current_user_count = log.log_user_count()

# --- Initial Setup ---
hide_close_button_css = """
    <style>
        div[aria-label="Modal"]>button[aria-label="Close"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_close_button_css, unsafe_allow_html=True)

# ---------- helper for guest button ----------
def guest_entry():
    st.session_state.user_logged_in = False
    st.session_state.show_welcome_modal = False

print("Welcome:",st.session_state.show_welcome_modal)
print("Logged in:",st.session_state.user_logged_in)

# ---------- welcome modal ----------
if st.session_state.show_welcome_modal and not st.session_state.show_login_modal:
    # hide close icon
    st.markdown(
        """
        <style>
        div[aria-label="Modal"]>button[aria-label="Close"] {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    welcome_modal = Modal("Welcome to SpectraGuru™", key="welcome_modal",
                  padding=20, max_width=600)

    with welcome_modal.container():
        st.info("SpectraGuru™ is still under development. Current version: SpectraGuru™ ver. 1.2.1")
        st.write("Thanks for visiting SpectraGuru™, a spectroscopy processing and visualization tool.")
        st.write("If you encounter a problem, please email Fengbo.Ma@uga.edu")
        st.write("**:arrow_upper_left: After starting, go to ‘Data Upload’ in the sidebar to begin!**")

        if LOCAL_DEPLOY:
            st.button("Continue as Guest", on_click=guest_entry)
        else:
            col1, col2 = st.columns(2)

            # left: guest
            col1.button("Continue as Guest", on_click=guest_entry)

            # right: login via Clerk—just a link
            #signin_url = clerk_signin_url()           # already returns the full redirect URL
            #col2.link_button("Log in", signin_url,type="primary")

            col2.button("Log in here", on_click=login, type="primary")


        st.caption(
            "By clicking any button you agree with the "
            "[Policy and Disclaimer of SpectraGuru™]"
            "(https://fengboma.github.io/docs.spectraguru/docs/License-Policies-Disclaimers.html)"
        )

print("Login modal:", st.session_state.show_login_modal)
print("User decided:", st.session_state.user_decided)
# ---------- login modal ----------- #
if st.session_state.show_login_modal and not LOCAL_DEPLOY:

    def abort():
        st.session_state.show_login_modal = False

    @st.dialog("Log in to SpectraGuru™", width="small", dismissible=True, on_dismiss=abort)
    def login_dialog():
        left, center, right = st.columns([2, 90, 1])
        
        with center:
            user = clerk_component(key="login", action="login")

            if populate(user):
                print("POPULATED")
                st.rerun()

    login_dialog()

st.write("# SpectraGuru™ - A Spectra Analysis Application")
if LOCAL_DEPLOY:
    st.write("## Local deploy version")
# st.info('SpectraGuru is still under development. Current version: SpectraGuru ver. 0.15')

# ---------- greet authenticated users ----------
if st.session_state.user_logged_in:
    username = st.session_state.get('username', 'Guest')
    st.write(f"Welcome {username}! 👋")
# -------------

# current_user_count = function.log_user_count(st.session_state.log_file_path)
try:
    counts = function.read_counts(st.session_state.log_file_path)

    # col1, col2, col3 = st.columns(3)
    # col1.metric("Views", st.session_state.current_user_count, None)
    # col2.metric("Plots generated", counts['Plot_Generated'], None)
    # col3.metric("Spectra processed", counts['Spectra_Processed'], None)
except:
    pass

st.sidebar.success("Navigate to the Data Upload page above to start")

if st.session_state.user_logged_in:
    st.sidebar.button("Log out", on_click=logout, type='primary')
elif not st.session_state.show_welcome_modal and not LOCAL_DEPLOY:
    st.sidebar.button("Log in", on_click=login, type='primary')

st.markdown(
    """
    SpectraGuru™ is a spectra analysis application designed to provide user-friendly tools for processing and visualizing spectra, aimed at accelerating your research. It functions as a dashboard or a specialized tool within a Python environment, organized with various modular functions that allow users to process spectroscopy data in a pipeline.

    **Visit our [documentation](https://fengboma.github.io/docs.spectraguru/) for more information about SpectraGuru™.**

    ---

    """)

st.info("Check out our latest news and updates on SpectraGuru™!")

# Add expander to show the flyer
with st.expander("View SpectraGuru™ flyer"):
    st.image("element/Spectraguru flyer v2 oct26 (1).png", caption="SpectraGuru™ flyer", use_container_width=True)
st.divider()


col1, col2 = st.columns([2, 12])
if col1.button(label='Data upload page', key='switch_data_upload_page', type="primary"):
    st.switch_page("pages/2_Data_Upload.py")
col2.markdown(' :arrow_left: **Start by uploading your data on the Data Upload page**')

st.divider()

st.markdown(
    """
    <div style="
        padding: 1.25rem 1.4rem;
        border-radius: 18px;
        background: #F0F2F6;
        border: 1px solid rgba(15, 23, 42, 0.10);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin: 0.25rem 0 0.5rem 0;
    ">
        <div style="
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: rgba(4, 99, 7, 0.10);
            color: #046307;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.7rem;
        ">CITATION</div>
        <h3 style="margin: 0 0 0.45rem 0; color: #046307;">If You Find SpectraGuru Useful</h3>
        <p style="margin: 0 0 1rem 0; color: #1f2937; line-height: 1.55;">
            If SpectraGuru contributes to your research, please consider citing our work. The references below can be copied directly into your manuscript.
        </p>
        <div style="padding: 0.9rem 1rem; background: rgba(255, 255, 255, 0.72); border-radius: 14px; border: 1px solid rgba(15, 23, 42, 0.08); margin-bottom: 0.85rem;">
            <p style="margin: 0; line-height: 1.6;">
                Fengbo Ma, Jiaheng Cui, Amit Kumar, Yanjun Yang, Xianyan Chen, and Yiping Zhao. <em>Comprehensive Open-Source Ecosystem for Raman and SERS Spectroscopy: Introducing SpectraGuru</em>. Analytical Chemistry. <a href="https://pubs.acs.org/doi/10.1021/acs.analchem.5c07799" target="_blank">Read online</a>
            </p>
        </div>
        <div style="padding: 0.9rem 1rem; background: rgba(255, 255, 255, 0.72); border-radius: 14px; border: 1px solid rgba(15, 23, 42, 0.08);">
            <p style="margin: 0;  line-height: 1.6;">
                Fengbo Ma, Jiaheng Cui, Amit Kumar, Yanjun Yang, Jessica McCabe Hutcheson, Xianyan Chen, Haijian Sun, and Yiping Zhao. <em>SpectraGuru: a community-guided path toward scalable Raman and SERS analysis</em>. In <em>Biomedical Vibrational Spectroscopy 2026: Advances in Research and Industry</em>, vol. 13846, pp. 31-41. SPIE, 2026. <a href="https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13846/1384608/SpectraGuru--a-community-guided-path-toward-scalable-Raman-and/10.1117/12.3086068.short" target="_blank">Read online</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

col1, col3 = st.columns([2,3])

col1.markdown(
    """
    ### Features include

    #### Data upload page
    **Data input**
    - Manual upload for TXT, CSV, and TSV files
    - Multi-file upload with a shared Raman-shift axis
    - Multi-class upload with class labels
    - Database search and batch selection for signed-in users
    - Automatic interpolation and overlap-range cropping for multiple classes
    - Label editing for downstream classification
    - Optional data preview controls
""")


col1.markdown(
    """
    #### Processing page
    """)



col1.markdown(
    """
    **Processing**
    - Interpolation
    - Crop
    - Despike
        - Auto despike
        - Manual despike
    - Smoothing
        - Savitzky-Golay filter
        - 1D Fast Fourier Transform filter
        - Median filter
        - Wavelet denoising
    - Baseline removal
        - AirPLS
        - ModPoly
        - Gaussian-Lorentzian Fitting
        - SNIP
        - Asymmetric least squares (ALS)
    - Normalization
        - Normalize by area
        - Normalize by peak
        - Min-max normalization
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
    - Preview data
    - Interactive plotting
    - Fast mode plotting
    - Custom axis titles
    - CSV data export
    - PNG plot export
    """)

col3.image(r'element/SpectraGuru Welcome Page Flow Chart.png')

# st.markdown(
#     """
    
#     """)
# col1, col2,col3 = st.columns([1,1,3])
col1.markdown(
    """
    #### Analytics page
    - Average plot with original spectra
    - Confidence interval plot
    - Spectra derivation
    - Fast Fourier Transform (FFT) analysis
    - Correlation heatmap
    - Peak identification and stats
    - Hierarchically clustered heatmap
    - Principal components analysis (PCA)
    - t‑Distributed stochastic neighbor embedding (t‑SNE)
    - Random Forest (RF) classification
    - K-nearest neighbors (KNN) classification
    - Support vector machine (SVM) classification

    #### Toolbox page
    - Spectra simulation
    """)


st.markdown(
    """
    ---

    ### Visitor geographic map

    The geographic map visualizes where visitors come from based on their latitude and longitude, with color intensity representing the frequency at each location. Each point on the map is derived from our records, showing the geographical distribution of visits.
"""
)

import streamlit.components.v1 as components
p = open(r"element/traffic_heatmap.html")
components.html(p.read(), scrolling=True, height=550)

st.markdown(
    """
    ---

    ### Function Usage

    The following table depicts the relative popularity of each of SpectraGuru™'s featured processing and analysis functions.
"""
)

if 'function_count_data' not in st.session_state or not log.CULL_FUNCTION_TABLE_REFRESH:
    st.session_state.function_count_data = log.get_count_data()

#st.dataframe(data=st.session_state.function_count_data, 
#                column_config={
#                    "Feature":st.column_config.TextColumn(width=200),
#                    "Algorithm":st.column_config.TextColumn(width=200),
#                    "Usage (Times Called)":st.column_config.NumberColumn(width=150),
#                    "References":st.column_config.TextColumn(width=100)
#                })
st.table(data=st.session_state.function_count_data)

st.markdown(
    """
    ---

    ### About Us 

    - Find us here: [Zhao Nano Lab](https://www.zhao-nano-lab.com/)
    - Explore more or report an issue? Send us a message to us (zhao-nano-lab@uga.edu)
"""
)

def image_data_uri(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

usda_logo = image_data_uri("element/USDA.png")
nsf_logo = image_data_uri("element/nsf.png")

st.markdown(
    f"""
    ### SpectraGuru™ supported by:
    <div style="display:flex; justify-content:center; align-items:center; gap:40px; flex-wrap:wrap;">
        <img src="{usda_logo}" style="height:80px; width:auto; max-width:320px; object-fit:contain;">
        <img src="{nsf_logo}" style="height:80px; width:auto; max-width:320px; object-fit:contain;">
    </div>
    """,
    unsafe_allow_html=True
)

st.session_state.global_placeholder = st.empty() # this should stay the last line of code in this file.
