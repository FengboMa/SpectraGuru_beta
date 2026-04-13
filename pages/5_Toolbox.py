import streamlit as st
import altair as alt
import pandas as pd
import function

function.wide_space_default()

DEFAULT_X_AXIS_TITLE = "Raman shift/cm⁻¹"
DEFAULT_Y_AXIS_TITLE = "Intensity/a.u."

st.sidebar.write("### SpectraGuru Toolbox")

st.sidebar.selectbox('Select Tool',
                     options=(
                         "Spectra Simulation"
                     ),
                     key="tool_select")

if st.session_state.tool_select == "Spectra Simulation":
    # Display parameter select interface
    st.sidebar.selectbox('Select Spectra Structure',
                         options=(
                             "Distinct",
                             "Joint",
                             "Consecutive"
                         ),
                         help="Distinct: All peaks are separated. Joint: Peaks are joined in pairs. Consecutive: Multiple peaks appear overlapping each other.",
                         key="simulation_structure_select")
    st.sidebar.number_input('Number of Spectra to generate', min_value=1, max_value=1000, value=1, key="simulation_batch_size_select")
    st.sidebar.button('Generate Spectra', key="simulation_button", type="primary")


st.write("## Toolbox")

if st.session_state.tool_select == "Spectra Simulation":
    st.write("##### Spectra Simulation")
    if 'simulation_df' not in st.session_state and not st.session_state.simulation_button:
        st.write("Click \"Generate Spectra\" to simulate random spectra.")
    else: 
        if st.session_state.simulation_button:
            st.session_state.simulation_df = pd.DataFrame() #function.generate_spectra()
        
        simulation_df = st.session_state.simulation_df
        simulated_spectra = alt.Chart(simulation_df).mark_line().encode(
                            x=alt.X('Ramanshift', title=DEFAULT_X_AXIS_TITLE, type="quantitative"),
                            y=alt.Y('Intensity', title=DEFAULT_Y_AXIS_TITLE, type="quantitative"),
                            color="Sample ID:N",
                            size=alt.value(1) # Line width
                        ).properties(
                            width=1300,
                            height=600,
                            title="Simulated Spectra"
                        )
        simulated_spectra = function.style_altair_chart(simulated_spectra)
        st.altair_chart(simulated_spectra, width="content")
