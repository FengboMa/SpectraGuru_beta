import streamlit as st
import altair as alt
import pandas as pd
import function
import log_utils as log

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
    st.sidebar.number_input("Number of Spectra to Generate", min_value=1, max_value=50, value=1, key="simulation_batch_size_select")
    
    st.sidebar.selectbox("Select Spectra Structure",
                            options=(
                                "Distinct",
                                "Joint",
                                "Consecutive"
                            ),
                            key="simulation_structure_select",
                            help="Distinct: All peaks are separated. Joint: Peaks are joined in pairs. Consecutive: Multiple peaks appear overlapping each other."
                        )
    # Special parameters
    if st.session_state.simulation_structure_select == "Distinct":
        st.sidebar.number_input("Average Number of Peaks", min_value=1, max_value=20, value=3, key="simulation_peak_number_select")
        st.sidebar.number_input("Peak Number Variance", min_value=0, max_value=20, value=0, key="simulation_peak_number_variance_select", help="The maximum variation in the number of peaks, centered at the 'Average Number of Peaks'.")
        st.sidebar.number_input("Separation Factor", min_value=1.0, max_value=10.0, value=3.0, step=0.1, key="simulation_separation_factor_select", help="Specifies the extent to which peak centers should be separated at a minimum. Warning: if set too high, some peaks may be forced to disobey the rule.")
    elif st.session_state.simulation_structure_select == "Joint":
        st.sidebar.number_input("Average Number of Regions", min_value=1, max_value=10, value=2, key="simulation_region_number_select", help="A region refers to a conjoined pair of peaks.")
        st.sidebar.number_input("Region Number Variance", min_value=0, max_value=10, value=1, key="simulation_region_number_variance_select", help="The maximum variation in the number of regions, centered at the 'Average Number of Regions'.")
        st.sidebar.number_input("Clustering Factor", min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="simulation_joint_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")
    elif st.session_state.simulation_structure_select == "Consecutive":
        st.sidebar.number_input("Average Peaks per Region", min_value=1, max_value=10, value=6, key="simulation_average_peaks_per_region_select", help="A region refers to a conjoined series of peaks.")
        st.sidebar.number_input("Per Region Peak Variance", min_value=0, max_value=10, value=2, key="simulation_per_region_peak_variance_select", help="The maximum variation in the number of peaks per region, centered at the 'Average Peaks per Region'.")
        st.sidebar.number_input("Clustering Factor", min_value=0.05, max_value=1.0, value=0.4, step=0.05, key="simulation_consecutive_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")

    st.sidebar.button("Generate Spectra", key="simulation_button", type="primary")


st.write("## Toolbox")

if st.session_state.tool_select == "Spectra Simulation":
    st.write("##### Spectra Simulation")
    if 'simulation_df' not in st.session_state and not st.session_state.simulation_button:
        st.write("Click \"Generate Spectra\" to simulate random spectra.")
    else: 
        if st.session_state.simulation_button:
            structure = st.session_state.simulation_structure_select
            num_spectra = st.session_state.simulation_batch_size_select
            s_params = {} # Special parameters
            if st.session_state.simulation_structure_select == "Distinct":
                s_params = {
                    "average_num_peaks":st.session_state.simulation_peak_number_select,
                    "peak_num_variance":st.session_state.simulation_peak_number_variance_select,
                    "separation_factor":st.session_state.simulation_separation_factor_select
                }
            elif st.session_state.simulation_structure_select == "Joint":
                s_params = {
                    "average_num_regions":st.session_state.simulation_region_number_select,
                    "region_num_variance":st.session_state.simulation_region_number_variance_select,
                    "clustering_factor":st.session_state.simulation_joint_clustering_factor_select
                }
            elif st.session_state.simulation_structure_select == "Consecutive":
                s_params = {
                    "average_peaks_per_region":st.session_state.simulation_average_peaks_per_region_select,
                    "per_region_peak_variance":st.session_state.simulation_per_region_peak_variance_select,
                    "clustering_factor":st.session_state.simulation_consecutive_clustering_factor_select
                }
            st.session_state.simulation_df = function.generate_spectra(structure=structure, num_spectra=num_spectra, s_params=s_params) # Call generating function
            log.log_function_call("Toolbox_Spectra_Simulation", f_params={
                'structure':structure,
                'num_spectra':num_spectra
            })
        
        simulation_df = st.session_state.simulation_df
        simulated_spectra = alt.Chart(simulation_df).mark_line().encode(
                            x=alt.X('Ramanshift', title=DEFAULT_X_AXIS_TITLE, type="quantitative"),
                            y=alt.Y('Intensity', title=DEFAULT_Y_AXIS_TITLE, type="quantitative"),
                            color="Sample ID:N",
                            size=alt.value(2) # Line width
                        ).properties(
                            width=1300,
                            height=600,
                            title="Simulated Spectra"
                        )
        simulated_spectra = function.style_altair_chart(simulated_spectra)
        st.altair_chart(simulated_spectra, width="content")
