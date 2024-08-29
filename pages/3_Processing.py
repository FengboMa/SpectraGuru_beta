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

# Sidebar for processing
if 'df' not in st.session_state:
    st.sidebar.write(" ")
        
else:
    st.sidebar.markdown("""
                    ### Processing
                    
                    """)
    # try:
    # st.session_state.crop_min = st.session_state.df.iloc[:, 0].min()
    # st.session_state.crop_max = st.session_state.df.iloc[:, 0].max()
    # with st.sidebar.form(key="processing_form"):
        
    #     st.write("Inside the form")
        
    #     interpolation_ref_x = round(st.session_state.df.iloc[:, 0])
    #     if 'interpolation_act' not in st.session_state:
    #         st.session_state.interpolation_act = False

    #     interpolation_act = st.toggle("Interpolation", value=False, help="Use Interpolation to transfer and round Ramanshift to its closest Integer.", key='interpolation_act')
            
    #     submitted = st.form_submit_button("Submit")
    # Interpolation
    # st.sidebar.markdown("**Interpolation**")
    interpolation_ref_x = round(st.session_state.df.iloc[:, 0])
    if 'interpolation_act' not in st.session_state:
        st.session_state.interpolation_act = False

    interpolation_act = st.sidebar.toggle("Interpolation", value=False, help="Use Interpolation to transfer and round Ramanshift to its closest Integer.", key='interpolation_act')
    # st.sidebar.write(interpolation_ref_x)
    
    # crop
    # st.sidebar.markdown("**Crop**")
    # st.session_state.crop_min = st.session_state.df.iloc[:, 0].min()
    # st.session_state.crop_max = st.session_state.df.iloc[:, 0].max()
    # if 'crop_act' not in st.session_state:
    #     st.session_state.crop_act = False
    # crop_act = st.sidebar.toggle("Turn on Crop", value=False, help="Use Crop to select range.", key='crop_act')
    
    # if crop_act:
    #     st.sidebar.number_input("Crop min", 
    #                             min_value=st.session_state.df.iloc[:, 0].min(), 
    #                             max_value=st.session_state.df.iloc[:, 0].max(),
    #                             value=st.session_state.df.iloc[:, 0].min(),
    #                             step=1.00,
    #                             key="crop_min")
        
    #     st.sidebar.number_input("Crop max", 
    #                             min_value=st.session_state.df.iloc[:, 0].min(), 
    #                             max_value=st.session_state.df.iloc[:, 0].max(),
    #                             value=st.session_state.df.iloc[:, 0].max(),
    #                             step=1.00,
    #                             key="crop_max")
        # cropping  = st.sidebar.slider("Select range for Spectra",st.session_state.df.iloc[:, 0].min(), st.session_state.df.iloc[:, 0].max(),
        #         (st.session_state.crop_min, st.session_state.crop_max), help="Crop spectra to the desired step size.")
    cropping  = st.sidebar.slider("Select range for Spectra",st.session_state.df.iloc[:, 0].min(), st.session_state.df.iloc[:, 0].max(),
            (st.session_state.df.iloc[:, 0].min(), st.session_state.df.iloc[:, 0].max()), help="Crop spectra to the desired step size.")
    
    
    
    # Despike
    # st.sidebar.markdown("**Despike**")
    
    if 'despike_act' not in st.session_state:
        st.session_state.despike_act = False
    
    despike_act = st.sidebar.toggle("Despike", 
                                    value=False, 
                                    help="Despike(Threshold, Width) Auto-Despikes spectra with old despike script. Replaces regions of spectra which increase more than a (Threshold) over a specified (Scan Width) with a line.", 
                                    key='despike_act')
    
    if despike_act:
        # Add more functions to this selectbox if needed
        st.session_state.despike_threshold = st.sidebar.number_input(label="Despike threshold",
                                                        min_value = 0, max_value = 1000, value = 300,
                                                        step = 1, placeholder="Insert a number")
        
        st.session_state.despike_zap_length = st.sidebar.number_input(label="Despike zap length / window size",
                                                        min_value = 0, max_value = 100, value = 11,
                                                        step = 1, placeholder="Insert a number")
    
    
    # Smoothening
    # st.sidebar.markdown("**Smoothening**")
    
    if 'smoothening_act' not in st.session_state:
        st.session_state.smoothening_act = False
    
    smoothening_act = st.sidebar.toggle("Smoothening", 
                                    value=False, 
                                    help="Smoothening in spectra processing is a technique used to reduce noise and enhance the signal by averaging adjacent data points to produce a clearer representation of the spectral data.", 
                                    key='smoothening_act')
    
    if smoothening_act:
        # Add more functions to this selectbox if needed
        smoothening_function = st.sidebar.selectbox(label="Select your smoothening Function",  options=["Savitzky-Golay filter","1D Fast Fourier Transform filter"])
        
        if smoothening_function == "Savitzky-Golay filter":
        # Add more functions to this selectbox if needed
            smoothening_act_window_length = st.sidebar.number_input(label="Savitzky-Golay window length",
                                                            min_value = 1, max_value = 100, value = 15,
                                                            step = 1, placeholder="Insert a number", help='The length of the filter window (i.e., the number of coefficients).')
            
            smoothening_act_polyorder = st.sidebar.number_input(label="Savitzky-Golay polynomial order",
                                                            min_value = 1, max_value = 15, value = 2,
                                                            step = 1, placeholder="Insert a number", help="The order of the polynomial used to fit the samples. polyorder must be less than Savitzky-Golay window length.")
        elif smoothening_function == "1D Fast Fourier Transform filter":
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
            smoothening_act_FFT_threshold = st.sidebar.number_input(label="FFT threshold",
                                                            min_value = 0.001, max_value = 10.000, value = 0.100
                                                            , placeholder="Insert a number", help=help_txt)

            help_txt2 = '''
            The padding_method parameter specifies the method used to pad the signal before applying the FFT. Padding helps to reduce edge effects and minimize artifacts introduced by the filtering process.
            
            **Mirror Padding ('mirror'):** Reflects the signal at its edges, creating a smooth transition.
            
            **Edge Padding ('edge'):** Repeats the edge values of the signal.
            
            **Zero Padding ('zero'):** Adds zeros to the edges of the signal. May introduce artifacts at the edges.
            '''
            smoothening_act_FFT_padding = st.sidebar.selectbox(label="Select your FFT Padding method",  options=["mirror",
                                                                                                                "edge",
                                                                                                                "zero"],
                                                            key = "smoothening_act_FFT_padding",
                                                            help = help_txt2)
    # Baseline removal
    # st.sidebar.markdown("**Baseline Removal**")
    
    if 'baselineremoval_act' not in st.session_state:
        st.session_state.baselineremoval_act = False
    
    baselineremoval_act = st.sidebar.toggle("Baseline Removal", 
                                            value=False, 
                                            help="Remove baselines (or backgrounds) from data by either by including a baseline function when fitting a sum of functions to the data, or by actually subtracting a baseline estimate from the data.", 
                                            key='baselineremoval_act')
    
    if baselineremoval_act:
        # Add more functions to this selectbox if needed
        baselineremoval_function = st.sidebar.selectbox(label="Select your Baseline Removal Function",  options=["airPLS", "ModPoly"])
        
        if baselineremoval_function == "airPLS":
            baselineremoval_airPLS_lambda = st.sidebar.number_input(label="AirPLS lambda", help="The larger lambda is,  the smoother the resulting background, z.",
                                                                    min_value = 1, max_value = 1000, value = 100,
                                                                    step = 1, placeholder="Insert a number")
            
            baselineremoval_airPLS_porder = st.sidebar.number_input(label="AirPLS p order", help="Adaptive iteratively reweighted penalized least squares for baseline fitting.",
                                                                    min_value=1, max_value = 10, value = 1, 
                                                                    step = 1, placeholder="Insert a number")
            
            baselineremoval_airPLS_itermax = st.sidebar.number_input(label="AirPLS max iteration",
                                                                    min_value=5, max_value = 100, value = 15, 
                                                                    step = 5, placeholder="Insert a number") 
        elif baselineremoval_function == "ModPoly":
            baselineremoval_ModPoly_degree = st.sidebar.number_input(label="ModPoly Polynomial degree",
                                                                    min_value=1, max_value = 20, value = 5, 
                                                                    step = 1, placeholder="Insert a number") 
    # Normalization
    # st.sidebar.markdown("**Normalization**")
    
    if 'normalization_act' not in st.session_state:
        st.session_state.normalization_act = False
    
    normalization_act = st.sidebar.toggle("Normalization", 
                                            value=False, 
                                            help="Normalize By Area(Area) simply divides each spectra's values by the area under the spectra then multiplies by the (Area) value. ie: It sets the area under each spectra equal to (Area)", 
                                            key='normalization_act')
    
    if normalization_act:
        # Add more functions to this selectbox if needed
        normalization_function = st.sidebar.selectbox(label="Select your Normalization Function",  options=["Normalize by area", 
                                                                                                            "Normalize by peak",
                                                                                                            "Min max normalize"])
    
    # Outlier Removal
    if 'outlierremoval_act' not in st.session_state:
        st.session_state.outlierremoval_act = False
    
    outlierremoval_act = st.sidebar.toggle("Outlier Removal", 
                                    value=False, 
                                    help="The function removes outlier spectra from a dataframe based on single threshold, distance, and correlation criteria.", 
                                    key='outlierremoval_act')
    
    if outlierremoval_act:
        # Add more functions to this selectbox if needed
        outlierremoval_act_single_threshold = st.sidebar.number_input(label="Outlier Removal Single Threshold",
                                                        min_value = 1, max_value = 20, value = 4,
                                                        step = 1, placeholder="Insert a number")
        
        outlierremoval_act_distance_threshold = st.sidebar.number_input(label="Outlier Removal Distance Threshold",
                                                        min_value = 1, max_value = 20, value = 6,
                                                        step = 1, placeholder="Insert a number")
        
        outlierremoval_act_correlation_threshold = st.sidebar.number_input(label="Outlier Removal correlation Threshold",
                                                        min_value = 1, max_value = 20, value = 4,
                                                        step = 1, placeholder="Insert a number")
    # """
    #     # Processing button and reaction
    #     if st.sidebar.button("Process", type='primary', key = 'process'):        
    #         function.log_spectra_processed_count(st.session_state.log_file_path)
    #         # interpolation act
    #         if interpolation_act:
    #             interpolated_df = pd.DataFrame(interpolation_ref_x, columns=[st.session_state.df.columns[0]])
    #             for col in st.session_state.df[1:]:
    #                 f = interp1d(st.session_state.df.iloc[:, 0], st.session_state.df[col], kind='linear',bounds_error=False,fill_value="extrapolate")
    #                 interpolated_values = f(interpolation_ref_x)
    #                 interpolated_df[col] = interpolated_values
    #             interpolated_df = interpolated_df.drop_duplicates()
    #             # st.sidebar.write(interpolated_df)
    #             st.session_state.df = interpolated_df
                
    #         # crop act
    #         st.session_state.df = st.session_state.df[(st.session_state.df.iloc[:, 0] >= cropping[0]) & (st.session_state.df.iloc[:, 0] <= cropping[1])]
            
    #         # despike_act
    #         if despike_act:
    #             st.session_state.df.iloc[:, 1:] = function.despikeSpec(spectra = st.session_state.df.iloc[:, 1:],
    #                                                                     ramanshift = st.session_state.df.iloc[:, 0],
    #                                                                     threshold = despike_act_threshold,
    #                                                                     zap_length = despike_act_zap_length)
            
            
    #         # smoothening_act
    #         if smoothening_act:
    #             if smoothening_function == "Savitzky-Golay filter":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply( lambda col: function.savgol_filter_spectra(col, 
    #                                                                                                                                 window_length = smoothening_act_window_length,
    #                                                                                                                                 polyorder = smoothening_act_polyorder))
            
    #             elif smoothening_function == "1D Fast Fourier Transform filter": 
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply( lambda col: function.FFT_spectra(col, 
    #                                                                                                                         FFT_threshold = smoothening_act_FFT_threshold,
    #                                                                                                                         padding_method = smoothening_act_FFT_padding))
                
    #         # baselineremoval_act
    #         if baselineremoval_act:
    #             if baselineremoval_function == "airPLS":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply( lambda col: function.airPLS(col.values, 
    #                                                                                                                             lambda_=baselineremoval_airPLS_lambda, 
    #                                                                                                                             porder=baselineremoval_airPLS_porder, 
    #                                                                                                                             itermax=baselineremoval_airPLS_itermax))
    #             if baselineremoval_function == "ModPoly":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:] - st.session_state.df.iloc[:, 1:].apply( lambda col: function.ModPoly(col.values, 
    #                                                                                                                             degree=baselineremoval_ModPoly_degree))
    #         # normalization act
    #         if normalization_act:
    #             if normalization_function == "Normalize by area":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.normalize_by_area, ramanshift =st.session_state.df.iloc[:, 0], axis = 0)        
    #             elif normalization_function == "Normalize by peak":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.normalize_by_peak, axis = 0)
    #             elif normalization_function == "Min max normalize":
    #                 st.session_state.df.iloc[:, 1:] = st.session_state.df.iloc[:, 1:].apply(function.min_max_normalize, axis = 0)
                    

    #         # outlier removal act
    #         if outlierremoval_act:
    #             df_cleaned, st.session_state.remove_outliers_log = function.remove_outliers(st.session_state.df)
    #             st.session_state.df = pd.concat([st.session_state.df.iloc[:, 0], df_cleaned], axis=1)
    # """


    @st.cache_data
    def process_data(
        df, 
        interpolation_act, 
        despike_act, 
        smoothening_act, 
        baselineremoval_act, 
        normalization_act, 
        outlierremoval_act,
        parameters):
        # Create a copy of the original dataframe to avoid modifying the original data
        processed_df = df.copy()

        # Interpolation
        if interpolation_act:
            interpolation_ref_x = round(processed_df.iloc[:, 0])
            interpolated_df = pd.DataFrame(interpolation_ref_x, columns=[processed_df.columns[0]])
            for col in processed_df.columns[1:]:
                f = interp1d(processed_df.iloc[:, 0], processed_df[col], kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated_values = f(interpolation_ref_x)
                interpolated_df[col] = interpolated_values
            processed_df = interpolated_df.drop_duplicates()

        # Cropping (if needed based on UI slider inputs)
        if 'crop_range' in parameters:
            crop_min, crop_max = parameters['crop_range']
            processed_df = processed_df[(processed_df.iloc[:, 0] >= crop_min) & (processed_df.iloc[:, 0] <= crop_max)]

        # Despiking
        if despike_act:
            processed_df.iloc[:, 1:] = function.despikeSpec(
                spectra=processed_df.iloc[:, 1:],
                ramanshift=processed_df.iloc[:, 0],
                threshold=parameters['despike_threshold'],
                zap_length=parameters['despike_zap_length']
            )

        # Smoothing
        if smoothening_act:
            if parameters['smoothening_function'] == "Savitzky-Golay filter":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:].apply(
                    lambda col: function.savgol_filter_spectra(
                        col, 
                        window_length=parameters['smooth_window_length'],
                        polyorder=parameters['smooth_polyorder']
                    )
                )
            elif parameters['smoothening_function'] == "1D Fast Fourier Transform filter":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:].apply(
                    lambda col: function.FFT_spectra(
                        col, 
                        FFT_threshold=parameters['FFT_threshold'],
                        padding_method=parameters['FFT_padding']
                    )
                )

        # Baseline Removal
        if baselineremoval_act:
            if parameters['baselineremoval_function'] == "airPLS":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:] - processed_df.iloc[:, 1:].apply(
                    lambda col: function.airPLS(
                        col.values, 
                        lambda_=parameters['airPLS_lambda'], 
                        porder=parameters['airPLS_porder'], 
                        itermax=parameters['airPLS_itermax']
                    )
                )
            elif parameters['baselineremoval_function'] == "ModPoly":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:] - processed_df.iloc[:, 1:].apply(
                    lambda col: function.ModPoly(
                        col.values, 
                        degree=parameters['ModPoly_degree']
                    )
                )

        # Normalization
        if normalization_act:
            if parameters['normalization_function'] == "Normalize by area":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:].apply(
                    function.normalize_by_area, 
                    ramanshift=processed_df.iloc[:, 0], 
                    axis=0
                )
            elif parameters['normalization_function'] == "Normalize by peak":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:].apply(
                    function.normalize_by_peak, 
                    axis=0
                )
            elif parameters['normalization_function'] == "Min max normalize":
                processed_df.iloc[:, 1:] = processed_df.iloc[:, 1:].apply(
                    function.min_max_normalize, 
                    axis=0
                )

        # Outlier Removal
        if outlierremoval_act:
            df_cleaned, outliers_log = function.remove_outliers(processed_df)
            processed_df = pd.concat([processed_df.iloc[:, 0], df_cleaned], axis=1)
            st.session_state.remove_outliers_log = outliers_log  # Storing the log in session state

        return processed_df

    
    # Usage Example in your Streamlit app
    if st.sidebar.button("Process", type='primary', key='process'):
        # Define parameters for processing
        parameters = {
            'despike_threshold': st.session_state.despike_threshold,
            'despike_zap_length': st.session_state.despike_zap_length,
            'smooth_window_length': st.session_state.smooth_window_length,
            'smooth_polyorder': st.session_state.smooth_polyorder,
            'smoothening_function': st.session_state.smoothening_function,
            'FFT_threshold': st.session_state.FFT_threshold,
            'FFT_padding': st.session_state.smoothening_act_FFT_padding,
            'baselineremoval_function': st.session_state.baselineremoval_function,
            'airPLS_lambda': st.session_state.baselineremoval_airPLS_lambda,
            'airPLS_porder': st.session_state.baselineremoval_airPLS_porder,
            'airPLS_itermax': st.session_state.baselineremoval_airPLS_itermax,
            'ModPoly_degree': st.session_state.baselineremoval_ModPoly_degree,
            'normalization_function': st.session_state.normalization_function,
            'crop_range': cropping  # Assuming cropping is defined based on the slider input
        }

        # Call the process_data function
        processed_data = process_data(
            st.session_state.df,
            interpolation_act=st.session_state.interpolation_act,
            despike_act=st.session_state.despike_act,
            smoothening_act=st.session_state.smoothening_act,
            baselineremoval_act=st.session_state.baselineremoval_act,
            normalization_act=st.session_state.normalization_act,
            outlierremoval_act=st.session_state.outlierremoval_act,
            parameters=parameters
        )

        # Store the processed data back in session state
        st.session_state.processed_df = processed_data
    # Function to reset the toggle
    # def reset_processing():
    #     st.session_state.df = st.session_state.df_original
    #     st.session_state.interpolation_act = False
    
    # Reset button and reaction
    if st.sidebar.button("Reset", type='secondary', on_click=function.reset_processing, key = 'reset'):
        st.session_state.df = st.session_state.backup
    # except:
    #     pass

# """"""""
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
    
    # Data muli selection
    st.session_state.df.rename(columns={st.session_state.df.columns[0]: 'Ramanshift'}, inplace=True)
    x_axis = st.session_state.df.columns[0]
    
    st.session_state.spectra_selected = st.multiselect(label= '**Select Spectra/s you want to visualize**', options= list(st.session_state.df.columns[1:]), default=list(st.session_state.df.columns[1:]))
    columns_to_select  = [st.session_state.df.columns[0]] + st.session_state.spectra_selected
    
    # Plot mode Select
    plot_row = row([0.1, 0.9])
    
    if plot_row.button("Plot", type='primary',key = 'plot') or st.session_state.get('process') or st.session_state.get('reset'):
        st.session_state.temp = st.session_state.df[columns_to_select]

    # st.session_state.start_time = time.time()
    # st.write(st.session_state.temp)
    # if st.button('end time'):
    #     st.session_state.elapsed_time = time.time() - st.session_state.start_time
    #     st.write("Test time (ms):")
    #     st.write(int(st.session_state.elapsed_time * 1000))
        
    mode_option = plot_row.toggle(label = 'Activate Fast Mode Plotting', value = st.session_state['update_mode_option'],  key = 'mode_option', help = 'Enable Fast Mode Plotting for faster plotting times by sacrificing interactive functions. If you upload more than 20 spectra, Fast Mode will be activated automatically.')
        
    
    try:
        data_melted = st.session_state.temp.melt(id_vars=[x_axis], var_name='Sample ID', value_name='Intensity')
        # st.session_state.data_melt = data_melted
        if "Average" in data_melted['Sample ID'].values:
            data_melted = data_melted[data_melted['Sample ID'] != "Average"]
            
        # if mode_option == False:
            
        #     base = alt.Chart(data_melted).mark_line().encode(
        #         x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
        #         y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
        #         color='Sample ID:N',
        #         size=alt.condition(
        #             alt.datum['Sample ID'] == 'Average',
        #             alt.value(2),  # Line width for the "Average" sample
        #             alt.value(2)   # Line width for other samples
        #         ),
        #         strokeDash=alt.condition(
        #             alt.datum['Sample ID'] == 'Average',
        #             alt.value([8, 8]),  # Dash pattern for the "Average" sample
        #             alt.value([1, 0])   # Solid line for other samples
        #         )
        #         ).properties(
        #             width=1300,
        #             height=600,
        #             title='Spectra Data Plot'
        #         ).interactive()
            
        #     # Define the nearest selection
        #     # click = alt.selection_single(nearest=True, on='click')
        #     st.altair_chart(base, use_container_width=False)
        #     function.log_plot_generated_count(st.session_state.log_file_path)
            
        # elif mode_option == True:
        #     base = alt.Chart(data_melted).mark_line().encode(
        #         x=alt.X(x_axis, title='Raman shift/cm^-1', type='quantitative'),
        #         y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
        #         tooltip=alt.value(None),
        #         color='Sample ID:N',
        #         size=alt.condition(
        #             alt.datum['Sample ID'] == 'Average',
        #             alt.value(2),  # Line width for the "Average" sample
        #             alt.value(2)   # Line width for other samples
        #         ),
        #         strokeDash=alt.condition(
        #             alt.datum['Sample ID'] == 'Average',
        #             alt.value([8, 8]),  # Dash pattern for the "Average" sample
        #             alt.value([1, 0])   # Solid line for other samples
        #         )
        #         ).properties(
        #             width=1300,
        #             height=600,
        #             title='Spectra Data Plot'
        #         )
        @st.cache_data
        def create_plot(data, mode_option, x_axis):
            if mode_option == False:
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
            else:
                base = alt.Chart(data).mark_line().encode(
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
            return base
            
            
        plot = create_plot(data_melted, mode_option, x_axis)
        st.altair_chart(plot, use_container_width=False)

        # Define the nearest selection
        st.altair_chart(base, use_container_width=False)
        function.log_plot_generated_count(st.session_state.log_file_path)
        
        
        # Download 
        @st.cache_data
        def download_df(df):
            return df.to_csv(index = False).encode("utf-8")
        
        csv = download_df(st.session_state.df)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_file_name = f"data_{current_time}.csv"

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=download_file_name,
            mime="text/csv",
        )
        
        if st.session_state.outlierremoval_act:
            st.write("**Following spectra has been detect and removed by outlier removal function**")
            st.table(st.session_state.remove_outliers_log)
        
    except:
        pass