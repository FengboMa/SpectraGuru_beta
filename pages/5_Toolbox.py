import streamlit as st
import altair as alt
import pandas as pd
import function
import log_utils as log
import datetime as dt

function.wide_space_default()

DEFAULT_X_AXIS_TITLE = "Raman shift/cm⁻¹"
DEFAULT_Y_AXIS_TITLE = "Intensity/a.u."

with st.sidebar:

    st.write("### SpectraGuru Toolbox")

    st.selectbox('Select Tool',
                        options=(
                            "Spectra Simulation"
                        ),
                        key="tool_select")

    if st.session_state.tool_select == "Spectra Simulation":
        # Display parameter select interface
        st.number_input("Number of Spectra to Generate", min_value=1, max_value=50, value=1, key="simulation_batch_size_select")
        st.number_input("Scale", min_value=0.01, max_value=10000.0, value=1.0, step=1.0, key="simulation_scale_select")
        
        st.selectbox("Select Spectra Structure",
                                options=(
                                    "Distinct",
                                    "Joint",
                                    "Consecutive"
                                ),
                                key="simulation_structure_select",
                                help="Distinct: All peaks are separated. Joint: Peaks are joined in pairs. Consecutive: Multiple peaks appear overlapping each other."
                            )
        
        structure = st.session_state.simulation_structure_select

        # Special parameters
        if structure == "Distinct":
            st.number_input("Average Number of Peaks", min_value=1, max_value=20, value=3, key="simulation_peak_number_select")
            st.number_input("Peak Number Variance", min_value=0, max_value=20, value=0, key="simulation_peak_number_variance_select", help="The maximum variation in the number of peaks, centered at the 'Average Number of Peaks'.")
            st.number_input("Separation Factor", min_value=1.0, max_value=10.0, value=3.0, step=0.1, key="simulation_separation_factor_select", help="Specifies the extent to which peak centers should be separated at a minimum. Warning: if set too high, some peaks may be forced to disobey the rule.")
        elif structure == "Joint":
            st.number_input("Average Number of Regions", min_value=1, max_value=10, value=2, key="simulation_region_number_select", help="A region refers to a conjoined pair of peaks.")
            st.number_input("Region Number Variance", min_value=0, max_value=10, value=1, key="simulation_region_number_variance_select", help="The maximum variation in the number of regions, centered at the 'Average Number of Regions'.")
            st.number_input("Clustering Factor", min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="simulation_joint_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")
        elif structure == "Consecutive":
            st.number_input("Average Peaks per Region", min_value=1, max_value=10, value=6, key="simulation_average_peaks_per_region_select", help="A region refers to a conjoined series of peaks.")
            st.number_input("Per Region Peak Variance", min_value=0, max_value=10, value=2, key="simulation_per_region_peak_variance_select", help="The maximum variation in the number of peaks per region, centered at the 'Average Peaks per Region'.")
            st.number_input("Clustering Factor", min_value=0.05, max_value=1.0, value=0.4, step=0.05, key="simulation_consecutive_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")
        
        # Baseline parameters
        st.toggle("Use Baseline", key="simulation_use_baseline")

        use_baseline = st.session_state.simulation_use_baseline

        if use_baseline:
            st.selectbox("Baseline Type",
                                    options=(
                                        "Polynomial",
                                        "Exponential",
                                        "Gaussian",
                                        "Sigmoidal"
                                    ),
                                    key="simulation_baseline_select",
                                    help="The shape of the baseline to be used. The baseline is normalized such that the plot spans from x=-1 to x=1."
                                )
            
            baseline_type = st.session_state.simulation_baseline_select

            if baseline_type == "Polynomial":
                st.caption("**Polynomial Baseline:**")
                st.caption("B(x) = a0 + a1*(x) + a2*(x^2) + a3*(x^3) + a4*(x^4) + a5*(x^5)")

                sliders = [st.slider(f"a{i}", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key=f"simulation_baseline_polynomial_a{i}") for i in range(6)]
            elif baseline_type == "Exponential":
                st.caption("**Exponential Baseline:**")
                st.caption("B(x) = a*(e^(-b * (x - x0)^2)) + c*((x - x0)^2)")

                st.slider("Amplitude (a)", min_value=-2.0, max_value=2.0, value=1.0, key="simulation_baseline_exponential_amplitude")
                st.slider("Exponent (b)", min_value=-1.0, max_value=1.0, value=-0.25, key="simulation_baseline_exponential_exponent")
                st.slider("Quadratic Term (c)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_exponential_quadratic_term")
                st.slider("Offset (x0)", min_value=-1.0, max_value=1.0, value=-0.5, key="simulation_baseline_exponential_offset")
            elif baseline_type == "Gaussian":
                st.caption("**Gaussian Baseline:**")
                st.caption("B(x) = a * (e^(-(x - mu)^2 / (2 * sigma^2)))")

                st.slider("Amplitude (a)", min_value=-1.0, max_value=1.0, value=0.5, key="simulation_baseline_gaussian_amplitude")
                st.slider("Center (mu)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_gaussian_center")
                st.slider("Width (sigma)", min_value=0.1, max_value=1.0, value=0.5, key="simulation_baseline_gaussian_width")
            elif baseline_type == "Sigmoidal":
                st.caption("**Sigmoidal Baseline:**")
                st.caption("B(x) = a / (1 + e^(-k * (x - x0)))")

                st.slider("Amplitude (a)", min_value=-1.0, max_value=1.0, value=0.5, key="simulation_baseline_sigmoidal_amplitude")
                st.slider("Exponent (k)", min_value=-1.0, max_value=1.0, value=0.1, key="simulation_baseline_sigmoidal_exponent")
                st.slider("Offset (x0)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_sigmoidal_offset")
        
        st.toggle("Use Noise", key="simulation_use_noise")

        use_noise = st.session_state.simulation_use_noise

        if use_noise:
            st.slider("Noise Amplifier", min_value=0.1, max_value=20.0, value=3.0, step=0.01, key="simulation_noise_amplifier")

        st.button("Generate Spectra", key="simulation_button", type="primary")


st.write("## Toolbox")

if st.session_state.tool_select == "Spectra Simulation":
    st.write("##### Spectra Simulation")
    if 'simulation_df' not in st.session_state and not st.session_state.simulation_button:
        st.write("Click \"Generate Spectra\" to simulate random spectra.")
    else: 
        if st.session_state.simulation_button:
            s_params = {} # Special parameters
            if structure == "Distinct":
                s_params["average_num_peaks"] = st.session_state.simulation_peak_number_select
                s_params["peak_num_variance"] = st.session_state.simulation_peak_number_variance_select
                s_params["separation_factor"] = st.session_state.simulation_separation_factor_select
            elif structure == "Joint":
                s_params["average_num_regions"] = st.session_state.simulation_region_number_select
                s_params["region_num_variance"] = st.session_state.simulation_region_number_variance_select
                s_params["clustering_factor"] = st.session_state.simulation_joint_clustering_factor_select
            elif structure == "Consecutive":
                s_params["average_peaks_per_region"] = st.session_state.simulation_average_peaks_per_region_select
                s_params["per_region_peak_variance"] = st.session_state.simulation_per_region_peak_variance_select
                s_params["clustering_factor"] = st.session_state.simulation_consecutive_clustering_factor_select

            b_params = {} # Baseline parameters
            if use_baseline:
                baseline_type = st.session_state.simulation_baseline_select
                if baseline_type == "Polynomial":
                    for i in range(6):
                        b_params[f"a{i}"] = st.session_state[f"simulation_baseline_polynomial_a{i}"]
                elif baseline_type == "Exponential":
                    b_params["a"] = st.session_state.simulation_baseline_exponential_amplitude
                    b_params["b"] = st.session_state.simulation_baseline_exponential_exponent
                    b_params["c"] = st.session_state.simulation_baseline_exponential_quadratic_term
                    b_params["x0"] = st.session_state.simulation_baseline_exponential_offset
                elif baseline_type == "Gaussian":
                    b_params["amp"] = st.session_state.simulation_baseline_gaussian_amplitude
                    b_params["c"] = st.session_state.simulation_baseline_gaussian_center
                    b_params["w"] = st.session_state.simulation_baseline_gaussian_width
                elif baseline_type == "Sigmoidal":
                    b_params["a"] = st.session_state.simulation_baseline_sigmoidal_amplitude
                    b_params["k"] = st.session_state.simulation_baseline_sigmoidal_exponent
                    b_params["x0"] = st.session_state.simulation_baseline_sigmoidal_offset
            
            # Call generating function
            num_spectra = st.session_state.simulation_batch_size_select
            scale = st.session_state.simulation_scale_select

            noise_amplifier = None
            if use_noise:
                noise_amplifier = st.session_state.simulation_noise_amplifier

            baseline_type = None
            if use_baseline:
                baseline_type = st.session_state.simulation_baseline_select

            st.session_state.simulation_df = function.generate_spectra(structure=structure, 
                                                                        num_spectra=num_spectra, 
                                                                        scale=scale,
                                                                        s_params=s_params, 
                                                                        use_baseline=use_baseline, 
                                                                        baseline_type=baseline_type, 
                                                                        b_params=b_params,
                                                                        use_noise=use_noise,
                                                                        noise_amplifier=noise_amplifier)
            log.log_function_call("Toolbox_Spectra_Simulation", f_params={
                                    'structure':structure,
                                    's_params':s_params,
                                    'num_spectra':num_spectra,
                                    'scale':scale,
                                    'use_baseline':use_baseline,
                                    'baseline_type':baseline_type,
                                    'b_params':b_params,
                                    'use_noise':use_noise,
                                    'noise_amplifier':noise_amplifier
                                })
        
        simulation_df = st.session_state.simulation_df
        simulation_df_melted = simulation_df.melt(id_vars="Ramanshift", var_name="Sample ID", value_name="Intensity")
        #print("Melted", simulation_df_melted)
        simulated_spectra = alt.Chart(simulation_df_melted).mark_line().encode(
                            x=alt.X('Ramanshift', title=DEFAULT_X_AXIS_TITLE, type="quantitative"),
                            y=alt.Y('Intensity', title=DEFAULT_Y_AXIS_TITLE, type="quantitative"),
                            color="Sample ID:N",
                            size=alt.value(2), # Line width
                            tooltip=alt.value(None)
                        ).properties(
                            width=1300,
                            height=600,
                            title="Simulated Spectra"
                        )
        simulated_spectra = function.style_altair_chart(simulated_spectra)
        st.altair_chart(simulated_spectra, width="content")

        @st.cache_data
        def download_df(df):
            return df.to_csv(index = False).encode("utf-8")
        
        # st.write(stats_download_df)
        
        stats_download_df = download_df(simulation_df)
        current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_file_name = f"data_Simulated_{current_time}.csv"

        st.download_button(
            label="Download Simulated data as CSV",
            data=stats_download_df,
            file_name=download_file_name,
            mime="text/csv",
        )