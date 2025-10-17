import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import altair as alt
# from streamlit_extras.chart_container import chart_container
from streamlit_extras.row import row
from scipy.interpolate import interp1d
from datetime import datetime

import function

function.wide_space_default()
st.session_state.log_file_path = r"element/user_count.txt"
st.session_state.function_log_file_path = r"element/funct_count.txt"
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# Testing
# import time
# st.session_state.start_time = time.time()
# sidebar_icon = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v11\element\UGA_logo_ExtremeHoriz_FC_MARCM.png"
# st.logo(sidebar_icon, icon_image=sidebar_icon)

""""""""
# Sidebar for processing
if 'df' not in st.session_state:
    st.sidebar.write(" ")
        
else:
    @st.fragment()
    def pre_processing():
        st.markdown("""
                        ### Processing
                        
                        Select processing steps:
                        """)
        # try:
        # st.session_state.crop_min = st.session_state.df.iloc[:, 0].min()
        # st.session_state.crop_max = st.session_state.df.iloc[:, 0].max()
        
        # Interpolation
        # st.sidebar.markdown("**Interpolation**")
        st.session_state.interpolation_ref_x = round(st.session_state.df.iloc[:, 0])
        if 'interpolation_act' not in st.session_state:
            st.session_state.interpolation_act = False

        interpolation_act = st.toggle("Interpolation", value=False, help="Use Interpolation to transfer and round Ramanshift to its closest Integer.", key='interpolation_act')
        # st.sidebar.write(interpolation_ref_x)
        
        # crop
        # st.sidebar.markdown("**Crop**")
        c_crop_min = float(st.session_state.df.iloc[:, 0].min())
        c_crop_max = float(st.session_state.df.iloc[:, 0].max())
        if 'crop_act' not in st.session_state:
            st.session_state.crop_act = False
        crop_act = st.toggle("Crop", value=False, help="Use Crop to select range.", key='crop_act')
        
        if crop_act:
            st.write("Spectra range: " ,c_crop_min, " - ", c_crop_max)
            st.number_input("Crop min", 
                                    min_value=0.00, 
                                    max_value=9999.00,
                                    value=c_crop_min,
                                    step=1.00,
                                    key="crop_min")
            
            st.number_input("Crop max", 
                                    min_value=0.00, 
                                    max_value=9999.00,
                                    value=c_crop_max,
                                    step=1.00,
                                    key="crop_max")
            st.session_state.cropping  = st.slider("Select range for Spectra",float(st.session_state.df.iloc[:, 0].min()), float(st.session_state.df.iloc[:, 0].max()),
                    (st.session_state.crop_min, st.session_state.crop_max), help="Crop spectra to the desired step size.")
        
        # cropping  = st.sidebar.slider("Select range for Spectra",st.session_state.df.iloc[:, 0].min(), st.session_state.df.iloc[:, 0].max(),
        #         (st.session_state.df.iloc[:, 0].min(), st.session_state.df.iloc[:, 0].max()), help="Crop spectra to the desired step size.")
        
        
        
        # Despike
        # st.sidebar.markdown("**Despike**")
        
        if 'despike_act' not in st.session_state:
            st.session_state.despike_act = False
        
        despike_act = st.toggle("Despike", 
                                        value=False, 
                                        help="**Auto-Despike Method** - Automatically detects and corrects spikes across the entire spectrum. Regions where the signal exceeds a defined threshold within a specified scan width are replaced with linear interpolation. This method  may slightly alter the overall spectrum. \n \n **Manual Despike Method** - Allows users to define a specific window  where despiking is applied. Only spikes within this region are corrected, minimizing unintended effects on the rest of the spectrum.", 
                                        key='despike_act')
        
        if despike_act:
            
            st.session_state.despike_function = st.selectbox(label="Select your despike Function",  options=["Auto despike method","Manual despike method"])
            
            if st.session_state.despike_function == "Auto despike method":
                # Add more functions to this selectbox if needed
                st.session_state.despike_act_threshold = st.number_input(label="Despike threshold",
                                                                min_value = 0, max_value = 1000, value = 300,
                                                                step = 1, placeholder="Insert a number")
                
                st.session_state.despike_act_zap_length = st.number_input(label="Despike zap length / window size",
                                                                min_value = 0, max_value = 100, value = 11,
                                                                step = 1, placeholder="Insert a number")
            elif st.session_state.despike_function == "Manual despike method":
                # Add more functions to this selectbox if needed
                st.session_state.despike_act_threshold = st.number_input(label="Despike threshold",
                                                                min_value = 0, max_value = 1000, value = 300,
                                                                step = 1, placeholder="Insert a number")
                
                st.session_state.despike_act_zap_length = st.number_input(label="Despike zap length / window size",
                                                                min_value = 0, max_value = 100, value = 11,
                                                                step = 1, placeholder="Insert a number")
                wavenumber_min = float(st.session_state.df.iloc[:, 0].min())
                wavenumber_max = float(st.session_state.df.iloc[:, 0].max())
                st.session_state.despike_fitting_ranges = []
                with st.form("despike_fitting_range_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        start = st.number_input(f"Start of range", key=f"despike_start", step=1.0, format="%.2f",value=wavenumber_min)
                    with col2:
                        end = st.number_input(f"End of range", key=f"despike_end", step=1.0, format="%.2f", value=wavenumber_max)
                    st.session_state.despike_fitting_ranges.append((start, end))

                    submitted = st.form_submit_button("Apply fitting ranges")
                    if submitted:
                        clipped = False

                        # Clip start if needed
                        if start < wavenumber_min:
                            st.warning(f"Start value clipped from {start:.2f} to {wavenumber_min:.2f}")
                            start = wavenumber_min
                            clipped = True

                        # Clip end if needed
                        if end > wavenumber_max:
                            st.warning(f"End value clipped from {end:.2f} to {wavenumber_max:.2f}")
                            end = wavenumber_max
                            clipped = True

                        # Validate order
                        if start >= end:
                            st.error("Start value must be less than End value.")
                        else:
                            st.session_state.despike_fitting_ranges.append((start, end))
                            st.success(f"Fitting range ({start:.2f}, {end:.2f}) applied.")
        
        # Smoothening
        # st.sidebar.markdown("**Smoothening**")
        
        if 'smoothening_act' not in st.session_state:
            st.session_state.smoothening_act = False
        
        smoothening_act = st.toggle("Smoothening", 
                                        value=False, 
                                        help="Smoothening in spectra processing is a technique used to reduce noise and enhance the signal by averaging adjacent data points to produce a clearer representation of the spectral data.", 
                                        key='smoothening_act')
        
        if smoothening_act:
            # Add more functions to this selectbox if needed
            st.session_state.smoothening_function = st.selectbox(label="Select your smoothening Function",  options=["Savitzky-Golay filter","1D Fast Fourier Transform filter"])
            
            if st.session_state.smoothening_function == "Savitzky-Golay filter":
            # Add more functions to this selectbox if needed
                st.session_state.smoothening_act_window_length = st.number_input(label="Savitzky-Golay window length",
                                                                min_value = 1, max_value = 100, value = 15,
                                                                step = 1, placeholder="Insert a number", help='The length of the filter window (i.e., the number of coefficients).')
                
                st.session_state.smoothening_act_polyorder = st.number_input(label="Savitzky-Golay polynomial order",
                                                                min_value = 1, max_value = 15, value = 2,
                                                                step = 1, placeholder="Insert a number", help="The order of the polynomial used to fit the samples. polyorder must be less than Savitzky-Golay window length.")
            elif st.session_state.smoothening_function == "1D Fast Fourier Transform filter":
                help_txt = '''
                The threshold is a parameter sets the cutoff frequency for the low-pass filter applied to the spectra in the frequency domain. This threshold determines which frequency components are preserved and which are filtered out.
                
                The filtering process is governed by the following equation:
                $$
                \\text{FFT\\_filtered}(k) = \\begin{cases} 
                \\text{FFT}(\\text{signal})(k) & \\text{if } |\\text{freq}(k)| \\leq \\text{cutoff} \\\\ 
                0 & \\text{otherwise}
                \\end{cases}
                $$
                
                '''
                st.session_state.smoothening_act_FFT_threshold = st.number_input(label="FFT threshold",
                                                                min_value = 0.001, max_value = 10.000, value = 0.100
                                                                , placeholder="Insert a number", help=help_txt)

                help_txt2 = '''
                The padding_method parameter specifies the method used to pad the signal before applying the FFT. Padding helps to reduce edge effects and minimize artifacts introduced by the filtering process.
                
                **Mirror Padding ('mirror'):** Reflects the signal at its edges, creating a smooth transition.
                
                **Edge Padding ('edge'):** Repeats the edge values of the signal.
                
                **Zero Padding ('zero'):** Adds zeros to the edges of the signal. May introduce artifacts at the edges.
                '''
                smoothening_act_FFT_padding = st.selectbox(label="Select your FFT Padding method",  options=["mirror",
                                                                                                                    "edge",
                                                                                                                    "zero"],
                                                                key = "smoothening_act_FFT_padding",
                                                                help = help_txt2)
        # Baseline removal
        # st.markdown("**Baseline Removal**")
        
        if 'baselineremoval_act' not in st.session_state:
            st.session_state.baselineremoval_act = False
        
        baselineremoval_act = st.toggle("Baseline Removal", 
                                                value=False, 
                                                help="Remove baselines (or backgrounds) from data by either by including a baseline function when fitting a sum of functions to the data, or by actually subtracting a baseline estimate from the data.", 
                                                key='baselineremoval_act')
        
        if baselineremoval_act:
            # Add more functions to this selectbox if needed
            st.session_state.baselineremoval_function = st.selectbox(label="Select your Baseline Removal Function",  options=["airPLS", "ModPoly","Gaussian-Lorentzian Fitting"])
            
            if st.session_state.baselineremoval_function == "airPLS":
                st.session_state.baselineremoval_airPLS_lambda = st.number_input(label="AirPLS lambda", help="The larger lambda is,  the smoother the resulting background, z.",
                                                                        min_value = 1, max_value = 1000000, value = 100,
                                                                        step = 1, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_porder = st.number_input(label="AirPLS p order", help="Adaptive iteratively reweighted penalized least squares for baseline fitting.",
                                                                        min_value=1, max_value = 10, value = 1, 
                                                                        step = 1, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_itermax = st.number_input(label="AirPLS max iteration",
                                                                        min_value=5, max_value = 1000, value = 15, 
                                                                        step = 5, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_tau = st.number_input(label="AirPLS tolerance",
                                                                        min_value=0.0000000001, max_value = 0.100000000, value = 0.001000000, 
                                                                        step = 0.0000000001, placeholder="Insert a number",format="%.10f") 
                
            elif st.session_state.baselineremoval_function == "ModPoly":
                st.session_state.baselineremoval_ModPoly_degree = st.number_input(label="ModPoly Polynomial degree",
                                                                        min_value=1, max_value = 20, value = 5, 
                                                                        step = 1, placeholder="Insert a number") 
            elif st.session_state.baselineremoval_function == "Gaussian-Lorentzian Fitting":
                st.session_state.baselineremoval_GLF_num_range = st.number_input(label="Number of fitting range",
                                                                        min_value=2, max_value = 10, value = 2, 
                                                                        step = 1, placeholder="Insert a number") 
                wavenumber_min = float(st.session_state.df.iloc[:, 0].min())
                wavenumber_max = float(st.session_state.df.iloc[:, 0].max())
                fitting_ranges = []
                with st.form("glf_fitting_range_form"):
                    for i in range(st.session_state.baselineremoval_GLF_num_range):
                        col1, col2 = st.columns(2)
                        with col1:
                            start = st.number_input(f"Start of range {i+1}", key=f"glf_start_{i}", step=1.0, format="%.2f")
                        with col2:
                            end = st.number_input(f"End of range {i+1}", key=f"glf_end_{i}", step=1.0, format="%.2f")
                        fitting_ranges.append((start, end))

                    submitted = st.form_submit_button("Apply fitting ranges")

                    if submitted:
                        valid = True
                        cleaned_ranges = []

                        for i, (start, end) in enumerate(fitting_ranges):
                            try:
                                start = float(start)
                                end = float(end)
                            except ValueError:
                                st.warning(f"Range {i+1}: Start or End is not a number.")
                                valid = False
                                continue

                            # Sort each range to ensure start < end
                            start, end = min(start, end), max(start, end)

                            # Clip to dataset bounds and warn
                            clipped = False

                            if start < wavenumber_min:
                                st.warning(f"Range {i+1}: Start value clipped from {start:.2f} to {wavenumber_min:.2f}")
                                start = wavenumber_min
                                clipped = True
                            if end > wavenumber_max:
                                st.warning(f"Range {i+1}: End value clipped from {end:.2f} to {wavenumber_max:.2f}")
                                end = wavenumber_max
                                clipped = True

                            # Additional check: range is still valid after clipping
                            if start >= end:
                                st.warning(f"Range {i+1}: Invalid after clipping (start {start:.2f} >= end {end:.2f}). Skipping this range.")
                                valid = False
                                continue

                            cleaned_ranges.append((start, end))

                            # Sort all ranges by start value
                            cleaned_ranges.sort(key=lambda x: x[0])

                            # Check for overlap
                            for i in range(len(cleaned_ranges) - 1):
                                current_end = cleaned_ranges[i][1]
                                next_start = cleaned_ranges[i+1][0]
                                if next_start <= current_end:
                                    st.warning(f"Range {i+1} and {i+2} are overlapping ({cleaned_ranges[i]} and {cleaned_ranges[i+1]})")
                                    valid = False

                            if valid:
                                st.session_state.fitting_ranges =cleaned_ranges
                                # st.write(st.session_state.despike_fitting_ranges)
                                st.success(f"Saved {len(cleaned_ranges)} valid fitting ranges.")
        # Normalization
        # st.markdown("**Normalization**")
        
        if 'normalization_act' not in st.session_state:
            normalization_act = False
        
        normalization_act = st.toggle("Normalization", 
                                                value=False, 
                                                help="Normalize By Area(Area) simply divides each spectra's values by the area under the spectra then multiplies by the (Area) value. ie: It sets the area under each spectra equal to (Area)", 
                                                key='normalization_act')
        
        if normalization_act:
            # Add more functions to this selectbox if needed
            st.session_state.normalization_function = st.selectbox(label="Select your Normalization Function",  options=["Normalize by area", 
                                                                                                                "Normalize by peak",
                                                                                                                "Min max normalize"])
        
        # Outlier Removal
        if 'outlierremoval_act' not in st.session_state:
            outlierremoval_act = False
        
        outlierremoval_act = st.toggle("Outlier Removal", 
                                        value=False, 
                                        help="The function removes outlier spectra from a dataframe based on single threshold, distance, and correlation criteria.", 
                                        key='outlierremoval_act')
        
        if outlierremoval_act:
            # Add more functions to this selectbox if needed
            st.session_state.outlierremoval_act_single_threshold = st.number_input(label="Outlier Removal Single Threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 4.00,
                                                            step = 0.01, placeholder="Insert a number")
            
            st.session_state.outlierremoval_act_distance_threshold = st.number_input(label="Outlier Removal Distance Threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 6.00,
                                                            step = 0.01, placeholder="Insert a number")
            
            st.session_state.outlierremoval_act_correlation_threshold = st.number_input(label="Outlier Removal correlation Threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 4.00,
                                                            step = 0.01, placeholder="Insert a number")
    with st.sidebar:
        pre_processing()
            
    # Processing button and reaction
    if st.sidebar.button("Process", type='primary', key = 'process'):        
        function.log_spectra_processed_count(st.session_state.log_file_path)
        # interpolation act
        if st.session_state.interpolation_act:
            interpolated_df = pd.DataFrame(st.session_state.interpolation_ref_x, columns=[st.session_state.df.columns[0]])
            for col in st.session_state.df[1:]:
                f = interp1d(st.session_state.df.iloc[:, 0], st.session_state.df[col], kind='linear',bounds_error=False,fill_value="extrapolate")
                interpolated_values = f(st.session_state.interpolation_ref_x)
                interpolated_df[col] = interpolated_values
            interpolated_df = interpolated_df.drop_duplicates()
            # st.write(interpolated_df)
            st.session_state.df = interpolated_df
            
        # crop act
        if st.session_state.crop_act:
            st.session_state.df = st.session_state.df[(st.session_state.df.iloc[:, 0] >= st.session_state.cropping[0]) & (st.session_state.df.iloc[:, 0] <= st.session_state.cropping[1])]
        
        # despike_act
        if st.session_state.despike_act:
            if st.session_state.despike_function == "Auto despike method":
                st.session_state.df.iloc[:, 1:] = function.despikeSpec(spectra = st.session_state.df.iloc[:, 1:],
                                                                    ramanshift = st.session_state.df.iloc[:, 0],
                                                                    threshold = st.session_state.despike_act_threshold,
                                                                    zap_length = st.session_state.despike_act_zap_length)
                function.log_function_use_count(st.session_state.function_log_file_path, "Auto_Despike_Called")
            elif st.session_state.despike_function == "Manual despike method":
                st.session_state.df.iloc[:, 1:] = function.despikeSpec_v2(spectra = st.session_state.df.iloc[:, 1:],
                                                                        ramanshift = st.session_state.df.iloc[:, 0],
                                                                        threshold = st.session_state.despike_act_threshold,
                                                                        zap_length = st.session_state.despike_act_zap_length,window_start=st.session_state.despike_fitting_ranges[0][0],window_end=st.session_state.despike_fitting_ranges[0][1])
                function.log_function_use_count(st.session_state.function_log_file_path, "Manual_Despike_Called")
        
        # smoothening_act
        if st.session_state.smoothening_act:
            if st.session_state.smoothening_function == "Savitzky-Golay filter":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply( lambda col: function.savgol_filter_spectra(col, 
                                                                                                                                window_length = st.session_state.smoothening_act_window_length,
                                                                                                                                polyorder = st.session_state.smoothening_act_polyorder))
                function.log_function_use_count(st.session_state.function_log_file_path, "Savgol_Filter_Called")
        
            elif st.session_state.smoothening_function == "1D Fast Fourier Transform filter": 
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply( lambda col: function.FFT_spectra(col, 
                                                                                                                        FFT_threshold = st.session_state.smoothening_act_FFT_threshold,
                                                                                                                        padding_method = st.session_state.smoothening_act_FFT_padding))
                function.log_function_use_count(st.session_state.function_log_file_path, "FFT_Filter_Called")
            
        # baselineremoval_act
        if st.session_state.baselineremoval_act:
            if st.session_state.baselineremoval_function == "airPLS":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply( lambda col: function.airPLS(col.values, 
                                                                                                                            lambda_=st.session_state.baselineremoval_airPLS_lambda, 
                                                                                                                            porder=st.session_state.baselineremoval_airPLS_porder, 
                                                                                                                            itermax=st.session_state.baselineremoval_airPLS_itermax,
                                                                                                                            tau=st.session_state.baselineremoval_airPLS_tau))
                function.log_function_use_count(st.session_state.function_log_file_path, "Air_PLS_Called")
            if st.session_state.baselineremoval_function == "ModPoly":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply( lambda col: function.ModPoly(col.values, 
                                                                                                                            degree=st.session_state.baselineremoval_ModPoly_degree))
                function.log_function_use_count(st.session_state.function_log_file_path, "Mod_Poly_Called")
            # if st.session_state.baselineremoval_function == "Gaussian-Lorentzian Fitting":
            #     st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply( lambda col: function.GLF(col.values, 
            #                                                                                                                                         wavenumber=st.session_state.df.iloc[:, 0].values,
            #                                                                                                                                         fitting_ranges=st.session_state.fitting_ranges))
            try:
                if st.session_state.baselineremoval_function == "Gaussian-Lorentzian Fitting":
                    st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply(
                        lambda col: function.GLF(
                            col.values,
                            wavenumber=st.session_state.df.iloc[:, 0].values,
                            fitting_ranges=st.session_state.fitting_ranges
                        )
                    )
                    function.log_function_use_count(st.session_state.function_log_file_path, "Gaussian_Lorentzian_Fitting_Called")
            except AttributeError as e:
                if "fitting_ranges" in str(e):
                    st.error("⚠️ Please go to the sidebar and apply your fitting ranges before applying the Gaussian-Lorentzian Fitting.")
                else:
                    raise e
        # normalization act
        if st.session_state.normalization_act:
            if st.session_state.normalization_function == "Normalize by area":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.normalize_by_area, ramanshift =st.session_state.df.iloc[:, 0], axis = 0)        
                function.log_function_use_count(st.session_state.function_log_file_path, "Area_Normalization_Called", st.session_state.df.iloc[:, 1:].shape[1])
            elif st.session_state.normalization_function == "Normalize by peak":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.normalize_by_peak, axis = 0)
                function.log_function_use_count(st.session_state.function_log_file_path, "Peak_Normalization_Called", st.session_state.df.iloc[:, 1:].shape[1])
            elif st.session_state.normalization_function == "Min max normalize":
                st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.min_max_normalize, axis = 0)
                function.log_function_use_count(st.session_state.function_log_file_path, "Minmax_Normalization_Called", st.session_state.df.iloc[:, 1:].shape[1])
                

        # outlier removal act
        if st.session_state.outlierremoval_act:
            df_cleaned, st.session_state.remove_outliers_log = function.remove_outliers(st.session_state.df,
                single_thresh=st.session_state.outlierremoval_act_single_threshold,
                distance_thresh=st.session_state.outlierremoval_act_distance_threshold,
                coeff_thresh=st.session_state.outlierremoval_act_correlation_threshold)
            st.session_state.df = pd.concat([st.session_state.df.iloc[:, 0], df_cleaned], axis=1)
            function.log_function_use_count(st.session_state.function_log_file_path, "Remove_Outliers_Called")
        
    
    # Function to reset the toggle
    # def reset_processing():
    #     st.session_state.df = st.session_state.df_original
    #     st.session_state.interpolation_act = False
    
    # Reset button and reaction

    if st.sidebar.button("Reset", type='secondary', on_click=function.reset_processing, key = 'reset'):
            # st.session_state.df = st.session_state.backup
            pass

    # st.write(st.session_state.backup)
""""""""
# Main Page
st.write("## Visualization and Processing")

if 'df' not in st.session_state:
    st.error('Please go back to Data Upload and upload your data.')
else:
    
    preview_act = st.toggle("Preview Data")
    
    if preview_act:
    
        st.write("**Preview**")
        
        st.dataframe(st.session_state.df, hide_index=True)
        # st.table(st.session_state.df)
    
    # arr = np.random.normal(1, 1, size=100)
    # fig, ax = plt.subplots()
    # ax.hist(arr, bins=20)

    # st.pyplot(fig)
    # Melt the dataframe to a long format for Altair
    
    # Create an Altair plot
    # st.write("#### Visualization on Spectra")
    ###############################
    # Data muli selection
    # st.session_state.df.rename(columns={st.session_state.df.columns[0]: 'Ramanshift'}, inplace=True)
    # x_axis = st.session_state.df.columns[0]
    
    # st.session_state.spectra_selected = st.multiselect(label= '**Select Spectra/s you want to visualize**', options= list(st.session_state.df.columns[1:]), default=list(st.session_state.df.columns[1:]))
    # columns_to_select  = [st.session_state.df.columns[0]] + st.session_state.spectra_selected
    
    # # Plot mode Select
    
    
    # if plot_row.button("Plot", type='primary',key = 'plot') or st.session_state.get('process') or st.session_state.get('reset'):
    #     st.session_state.temp = st.session_state.df[columns_to_select]
###################################

    @st.cache_data
    def get_selected_columns(df, selected_spectra):
        columns_to_select = [df.columns[0]] + selected_spectra
        return df[columns_to_select]

    # Ensure first column is renamed (only once per session)
    if "df" in st.session_state:
        if st.session_state.df.columns[0] != "Ramanshift":
            st.session_state.df.rename(columns={st.session_state.df.columns[0]: "Ramanshift"}, inplace=True)

        x_axis = st.session_state.df.columns[0]  # This is now always 'Ramanshift'

        # Multiselect for spectra selection
        st.session_state.spectra_selected = st.multiselect(
            label="**Select Spectra/s you want to visualize**",
            options=list(st.session_state.df.columns[1:]),
            default=list(st.session_state.df.columns[1:])
        )

        # Get cached selected data
        plot_row = row([0.1, 0.9])
        if plot_row.button("Plot", type="primary", key="plot") or st.session_state.get("process") or st.session_state.get("reset"):
            st.session_state.temp = get_selected_columns(st.session_state.df, st.session_state.spectra_selected)

    
    # st.write(st.session_state.temp)
    # if st.button('end time'):
    #     st.session_state.elapsed_time = time.time() - st.session_state.start_time
    #     st.write("Test time (ms):")
    #     st.write(int(st.session_state.elapsed_time * 1000))
    # plot_row = row([0.1, 0.9])
    mode_option = plot_row.toggle(label = 'Activate Fast Mode Plotting', value = st.session_state['update_mode_option'],  key = 'mode_option', help = 'Enable Fast Mode Plotting for faster plotting times by sacrificing interactive functions. If you upload more than 20 spectra, Fast Mode will be activated automatically.')
        
    
    try:
        st.toast("Processing...", icon="📍")
        # data_melted = st.session_state.temp.melt(id_vars=[x_axis], var_name='Sample ID', value_name='Intensity')
        # # st.session_state.data_melt = data_melted
        # if "Average" in data_melted['Sample ID'].values:
        #     data_melted = data_melted[data_melted['Sample ID'] != "Average"]
        ################################
        @st.cache_data
        def melt_and_filter_data(temp_df, x_axis):
            """Caches the melted DataFrame and removes 'Average' if present."""
            data_melted = temp_df.melt(id_vars=[x_axis], var_name='Sample ID', value_name='Intensity')
            
            # Remove 'Average' if it exists
            if "Average" in data_melted['Sample ID'].values:
                data_melted = data_melted[data_melted['Sample ID'] != "Average"]
            
            return data_melted
        # @st.cache_data
        # def melt_and_filter_data(temp_df, x_axis):
        #     """Caches the melted DataFrame and removes 'Average' if present."""
        #     data_melted = temp_df.melt(id_vars=[x_axis], var_name='Sample ID', value_name='Intensity')
            
        #     # Remove 'Average' if it exists
        #     if "Average" in data_melted['Sample ID'].values:
        #         data_melted = data_melted[data_melted['Sample ID'] != "Average"]
            
        #     return data_melted

        # Ensure 'temp' exists in session state before calling
        # if "temp" in st.session_state:
        #     data_melted = melt_and_filter_data(st.session_state.temp, x_axis)

            # Store in session state if needed
            # st.session_state.data_melt = data_melted
        #########################################    
        if mode_option == False:
            # Original V10
            # # Create a selection object
            # nearest = alt.selection_single(
            #     nearest=True,
            #     on='mouseover',
            #     fields=[x_axis],
            #     empty='none',
            # )

            # base = alt.Chart(data_melted).mark_line().encode(
            #     x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
            #     y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
            #     color='Sample ID:N',
            #     size=alt.condition(
            #         alt.datum['Sample ID'] == 'Average',
            #         alt.value(4),  # Line width for the "Average" sample
            #         alt.value(2)   # Line width for other samples
            #     ),
            #     strokeDash=alt.condition(
            #         alt.datum['Sample ID'] == 'Average',
            #         alt.value([8, 8]),  # Dash pattern for the "Average" sample
            #         alt.value([1, 0])   # Solid line for other samples
            #     )
            # )
            # # Create selectors
            # selectors = alt.Chart(data_melted).mark_point().encode(
            #     x=alt.X(x_axis, type='quantitative'),
            #     opacity=alt.value(0)
            # ).add_selection(
            #     nearest
            # )

            # # Define points to be highlighted on hover
            # points = base.mark_point().transform_filter(
            #     nearest
            # )

            # text = alt.Chart(data_melted).mark_text(
            #     align='left',
            #     dx=5,
            #     dy=-5,
            #     fontSize=15,
            #     fontWeight=600,
            # ).transform_filter(
            #     nearest
            # ).encode(
            #     x=alt.X(x_axis, type='quantitative'),
            #     y=alt.Y('Intensity', type='quantitative'),
            #     text=alt.condition(nearest, 'Intensity:Q', alt.value(' ')),
            #     color='Sample ID:N'
            # )

            # # Create the rule for the vertical line
            # rules = alt.Chart(data_melted).mark_rule(color='gray').encode(
            #     x=alt.X(x_axis, type='quantitative')
            # ).transform_filter(
            #     nearest
            # )

            # # Layer the base, selectors, points, text, and rules
            # chart = alt.layer(
            #     base, selectors, points, rules, text
            # ).properties(
            #     width=1300,
            #     height=600,
            #     title='Spectra Data Plot'
            # ).interactive()
            
            # st.altair_chart(chart, use_container_width=False)
            
            
            # base = alt.Chart(data_melted).mark_line().encode(
            #     x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
            #     y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
            #     color='Sample ID:N',
            #     size=alt.condition(
            #         alt.datum['Sample ID'] == 'Average',
            #         alt.value(2),  # Line width for the "Average" sample
            #         alt.value(2)   # Line width for other samples
            #     ),
            #     strokeDash=alt.condition(
            #         alt.datum['Sample ID'] == 'Average',
            #         alt.value([8, 8]),  # Dash pattern for the "Average" sample
            #         alt.value([1, 0])   # Solid line for other samples
            #     )
            #     ).properties(
            #         width=1300,
            #         height=600,
            #         title='Spectra Data Plot'
            #     ).interactive()
            
            # # Define the nearest selection
            # # click = alt.selection_single(nearest=True, on='click')
            # st.altair_chart(base, use_container_width=False)
            


            @st.cache_data
            # Generate the Altair plot
            def generate_altair_plot(data):
                base = alt.Chart(data).mark_line().encode(
                x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
                y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
                color='Sample ID:N',
                size=alt.condition(
                    alt.datum['Sample ID'] == 'Average',
                    alt.value(2),  # Line width for the "Average" sample
                    alt.value(2)   # Line width for other samples
                ),
                strokeDash=alt.condition(
                    alt.datum['Sample ID'] == 'Average',
                    alt.value([8, 8]),  # Dash pattern for the "Average" sample
                    alt.value([1, 0])   # Solid line for other samples
                )
                ).properties(
                    width=1300,
                    height=600,
                    title='Spectra Data Plot'
                ).interactive()
                base = function.style_altair_chart(base)
                return base
            
            # st.write("xxxxx")
            
            if "temp" in st.session_state:
                data_melted = melt_and_filter_data(st.session_state.temp, x_axis)
            
            cached_plot = generate_altair_plot(data_melted)
            # st.write("yyyyy")
            # Display Plot
            st.altair_chart(cached_plot, use_container_width=False)
            # st.write("zzzzz")
            function.log_plot_generated_count(st.session_state.log_file_path)
            
        elif mode_option == True:
                
            @st.cache_data
            # Generate the Altair plot
            def generate_altair_plot_fastmode(data):
                base = base = alt.Chart(data).mark_line().encode(
                x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
                y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
                tooltip=alt.value(None),
                color='Sample ID:N',
                size=alt.condition(
                    alt.datum['Sample ID'] == 'Average',
                    alt.value(2),  # Line width for the "Average" sample
                    alt.value(2)   # Line width for other samples
                ),
                strokeDash=alt.condition(
                    alt.datum['Sample ID'] == 'Average',
                    alt.value([8, 8]),  # Dash pattern for the "Average" sample
                    alt.value([1, 0])   # Solid line for other samples
                )
                ).properties(
                    width=1300,
                    height=600,
                    title='Spectra Data Plot'
                )
                base = function.style_altair_chart(base)
                return base
            
            if "temp" in st.session_state:
                data_melted = melt_and_filter_data(st.session_state.temp, x_axis)
            
            cached_plot = generate_altair_plot_fastmode(data_melted)
            # st.write("1111")
            # Display Plot
            st.altair_chart(cached_plot, use_container_width=False)
            # st.write("2222")
            function.log_plot_generated_count(st.session_state.log_file_path)
            # # Define the nearest selection
            # # click = alt.selection_single(nearest=True, on='click')
            # st.altair_chart(base, use_container_width=False)
            # function.log_plot_generated_count(st.session_state.log_file_path)
        
        
        # Download 
        @st.cache_data
        def download_df(df):
            return df.to_csv(index = False).encode("utf-8")
        
        csv = download_df(st.session_state.df)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_file_name = f"data_{current_time}.csv"
        download_plot_name = f"spectra_plot_{current_time}.png"

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=download_file_name,
            mime="text/csv",
        )
        
        png_bytes = function.make_matplotlib_png(data_melted, x_axis)
        st.download_button(
            "Download plot (PNG, 600 dpi)",
            data=png_bytes,
            file_name=download_plot_name,
            mime="image/png",
            use_container_width=False,
    )
        
        if st.session_state.outlierremoval_act:
            st.write("**Following spectra has been detect and removed by outlier removal function**")
            st.table(st.session_state.remove_outliers_log)
        
    except:
        pass