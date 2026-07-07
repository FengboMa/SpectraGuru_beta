import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import altair as alt
# from streamlit_extras.chart_container import chart_container
from streamlit_extras.row import row
import threading
from datetime import datetime
import pandas as pd
import function
import log_utils as log

function.wide_space_default()

DEFAULT_X_AXIS_TITLE = "Raman shift/cm⁻¹"
DEFAULT_Y_AXIS_TITLE = "Intensity/a.u."

# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

if 'df' in st.session_state:
    st.sidebar.write("#### Analytics")

    # Side bar to select which plot and parameters
    st.sidebar.selectbox('Select analytics plot',
                        options= ("Average Plot with Original Spectra",
                                "Confidence Interval Plot",
                                "Spectral Derivation",
                                "Fast Fourier Transform (FFT) analysis",
                                "Correlation Heatmap",
                                "Peak Identification and Stats",
                                "Hierarchically-clustered Heatmap",
                                "Principal Components Analysis (PCA)",
                                "T-SNE Dimensionality Reduction",
                                "Random Forest(RF) Classification",
                                "K-Nearest Neighbors(KNN) Classification",
                                "Support Vector Machine(SVM) Classification",
                                "Full Spectrum Fitting"),
                        key="stats_plot_select")

    if st.session_state.stats_plot_select == "Average Plot with Original Spectra":
        st.sidebar.toggle(label='Show spectra you selected', value=True, key = 'stats_avg_act',help='Show or hide original selected spectra.')
        st.sidebar.toggle(label='Show standard deviation', value=True, key = 'stats_avg_std_act',help='Show or hide standard deviation.')
    elif st.session_state.stats_plot_select == "Confidence Interval Plot":
        # Interval method selector
        interval_method = st.sidebar.radio(
            label="Interval method",
            options=("Confidence Interval", "Standard Deviation"),
            index=0,   # default to Confidence Interval
            help=(
                "Choose how the uncertainty band is computed.\n\n"
                "Confidence Interval estimates the uncertainty of the mean using "
                "the t-distribution.\n\n"
                "Standard Deviation shows how replicate spectra vary from each other."
            ),
            key = "interval_method"
        )

        # If user selects Confidence Interval method
        if interval_method == "Confidence Interval":
            conf_lvl = st.sidebar.selectbox(
                label="Confidence level",
                options=(90, 95, 99),
                index=1,   # default to 95 percent
                key="conf_lvl",
                help=(
                    "Choose the confidence level for the interval. "
                    "Higher levels produce wider intervals. "
                    "The interval is computed using the formula: "
                    "$CI = \\bar{x} \\pm t \\cdot (s / \\sqrt{n})$, "
                    "where $\\bar{x}$ is the mean, $s$ is the standard deviation of replicates, "
                    "and $n$ is the number of spectra.\n\n"
                    "This interval estimates **uncertainty of the mean spectrum**, not the "
                    "spread of the raw spectra."
                )
            )

        else:
            # Standard deviation envelope
            std_multiplier = st.sidebar.selectbox(
                label="Number of standard deviations",
                options=(1, 2, 3),
                index=0,   # default to 1 SD
                key="std_mult",
                help=(
                    "Choose how many standard deviations to use when forming the envelope. "
                    "For example, 1 SD typically captures about 68 percent of spectra if data is "
                    "normally distributed.\n\n"
                    "This method visualizes **spread among individual spectra**, not the "
                    "uncertainty of the mean."
                )
            )
    
    elif st.session_state.stats_plot_select == "Spectral Derivation":
        st.sidebar.selectbox(label="Normalization method",
            options=("None", "Min-Max Normalization"),
            index=0,
            key="deriv_norm_method",
            help="Apply per-spectrum Min–Max scaling before taking derivatives.")
    elif st.session_state.stats_plot_select == "Fast Fourier Transform (FFT) analysis":
        fft_target_options = ["Average"] + [
            column for column in st.session_state.temp.columns
            if column not in ("Ramanshift", "Average", "Standard Deviation")
        ]
        st.sidebar.selectbox(
            label="Select spectrum for FFT",
            options=fft_target_options,
            index=0,
            key="fft_target_spectrum"
        )
        st.sidebar.toggle(
            label="Subtract average value before FFT",
            value=False,
            key="fft_subtract_average"
        )
    elif st.session_state.stats_plot_select == "Correlation Heatmap":

        st.sidebar.selectbox(
            label='Correlation algorithm',
            options=('Pearson Correlation', 'Cosine Similarity'),
            index=0,
            key='heatmap_corr_method',
            help=(
                "Choose the correlation/similarity algorithm:\n\n"
                "**Pearson Correlation**: Measures linear correlation between variables. "
                "Values range from -1 (perfect negative) to 1 (perfect positive). "
                "Sensitive to scale and magnitude.\n\n"
                "**Cosine Similarity**: Measures the cosine of the angle between vectors. "
                "Values range from -1 to 1. Less sensitive to magnitude, focuses on direction/shape."
            )
        )
        
        st.sidebar.toggle(label='Compute average', value=False, key='heatmap_compute_avg', help='Add an average row/column at the end of the heatmap.')

        if st.sidebar.toggle(label='Customize heatmap scale', value=False, key = 'heatmap_scale',help='Customize heatmap scale manually.'):
            st.sidebar.number_input(label='Heatmap scale min',min_value= -1.0, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_min',step = 0.01, value = 0.5, format="%.2f")
            st.sidebar.number_input(label='Heatmap scale max',min_value= -1.00, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_max',step = 0.01, value = 1.00, format="%.2f")
            
            if st.session_state.heatmap_min >= st.session_state.heatmap_max:
                st.sidebar.error('Invalid number input.')
    elif st.session_state.stats_plot_select == "Peak Identification and Stats":
        peak_target_options = ["Average"] + [
            column for column in st.session_state.temp.columns
            if column not in ["Ramanshift", "Average", "Standard Deviation"]
        ]
        st.sidebar.selectbox(
            label="Select spectrum for peak identification",
            options=peak_target_options,
            index=0,
            key="peak_identification_target"
        )
        
        st.sidebar.toggle(label="Auto peak identification", value=True, key="peak_iden_auto")
        
        if st.session_state.peak_iden_auto == True:
            st.session_state.peak_iden_height_p = 0
            st.session_state.peak_iden_threshold_p = 0
            st.session_state.peak_iden_distance_p = 1
            st.session_state.peak_iden_prominence_p = 0
            st.session_state.peak_iden_width_p = 0
            
            st.sidebar.number_input(label='Number of peaks to identify',min_value= 1, max_value= 10000, placeholder='Insert a number',
                                        key = 'peak_iden_auto_num',step = 1, value = 10,
                                        help = "Number of peaks to identify automatically.")
            
        else:       
            st.sidebar.number_input(label='Height',min_value= 0.00, max_value= 10000.00, placeholder='Insert a number',
                                        key = 'peak_iden_height',step = 1.00, value = None,
                                        help = "Required height of peaks.")
            
            st.sidebar.number_input(label='Threshold',min_value= 0.00, max_value= 5000.00, placeholder='Insert a number',
                                        key = 'peak_iden_threshold',step = 1.00, value = None,
                                        help = "Required threshold of peaks, the vertical distance to its neighboring samples.")
            
            st.sidebar.number_input(label='Distance',min_value= 1.00, max_value= 5000.00, placeholder='Insert a number',
                                        key = 'peak_iden_distance',step = 1.00, value = None,
                                        help = "Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.")
            
            st.sidebar.number_input(label='Prominence',min_value= 0.00, max_value= 5000.00, placeholder='Insert a number',
                                        key = 'peak_iden_prominence',step = 1.00, value = None,
                                        help = "Required prominence of peaks. The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.")
            
            st.sidebar.number_input(label='Width',min_value= 0.00, max_value= 5000.00, placeholder='Insert a number',
                                        key = 'peak_iden_width',step = 1.00, value = None,
                                        help = "Required width of peaks in samples.")
            
            st.session_state.peak_iden_height_p = st.session_state.peak_iden_height
            st.session_state.peak_iden_threshold_p = st.session_state.peak_iden_threshold
            st.session_state.peak_iden_distance_p = st.session_state.peak_iden_distance
            st.session_state.peak_iden_prominence_p = st.session_state.peak_iden_prominence
            st.session_state.peak_iden_width_p = st.session_state.peak_iden_width
    elif st.session_state.stats_plot_select == "Hierarchically-clustered Heatmap":
        st.sidebar.toggle(label="Show clustered heatmap", value=True, key="HCA_heatmap")
    elif st.session_state.stats_plot_select == "Principal Components Analysis (PCA)":
        num_rows = st.session_state.df.shape[1] - 1
        pc_list = [f"PC{i+1}" for i in range(num_rows)] 
        st.sidebar.selectbox(label="Select horizontal PC", options=pc_list, index=0,key="PCA_horizontal")
        st.sidebar.selectbox(label="Select vertical PC", options=pc_list, index=1,key="PCA_vertical")
        st.sidebar.toggle(label="Color by labels", value=True, key="PCA_label")
    elif st.session_state.stats_plot_select == "T-SNE Dimensionality Reduction":
        max_perplexity = st.session_state.df.shape[1] - 1
        st.sidebar.select_slider(label="t-SNE Perplexity", options=list(range(1,max_perplexity)),value=2, key="tSNE_perplexity")
        st.sidebar.select_slider(label="t-SNE Maximum number of iterations", options=list(range(200,1001)), value=500, key="tSNE_n_iter")
    elif st.session_state.stats_plot_select == "Random Forest(RF) Classification":
        with st.sidebar.form("rf_classification_form"):
            st.number_input(
                label="Number of trees",
                min_value=1,
                max_value=500,
                step=1,
                value=100,
                key="rf_n_estimators",
                help="Number of decision trees in the forest. Default: 100.",
            )
            st.number_input(
                label="Maximum tree depth",
                min_value=0,
                max_value=100,
                step=1,
                value=0,
                key="rf_max_depth",
                help="Maximum number of splits per tree. Default: 0, meaning unlimited depth.",
            )
            st.number_input(
                label="Minimum samples per leaf",
                min_value=1,
                max_value=50,
                step=1,
                value=1,
                key="rf_min_samples_leaf",
                help="Minimum spectra required in a terminal tree leaf. Default: 1.",
            )
            st.slider(
                label="Test size (%)",
                min_value=0,
                max_value=80,
                step=1,
                value=0,
                key="rf_test_size",
                help="Percentage of spectra held out for evaluation. Default: 0%, which fits and evaluates on the full selected dataset.",
            )
            rf_run = st.form_submit_button("Run Random Forest")
    elif st.session_state.stats_plot_select == "K-Nearest Neighbors(KNN) Classification":
        with st.sidebar.form("knn_classification_form"):
            st.number_input(
                label="Number of neighbors (K)",
                min_value=1,
                max_value=100,
                step=1,
                value=3,
                key="knn_n_neighbors",
                help="Number of nearest training spectra used for voting. Default: 3.",
            )
            st.selectbox(
                label="Weights",
                options=("uniform", "distance"),
                index=0,
                key="knn_weights",
                help="Voting rule for neighbors. Default: uniform, giving each neighbor equal weight.",
            )
            st.selectbox(
                label="Distance metric",
                options=("euclidean", "manhattan", "minkowski"),
                index=0,
                key="knn_metric",
                help="Distance function used to identify nearest spectra. Default: euclidean.",
            )
            st.slider(
                label="Test size (%)",
                min_value=0,
                max_value=80,
                step=1,
                value=0,
                key="knn_test_size",
                help="Percentage of spectra held out for evaluation. Default: 0%, which fits and evaluates on the full selected dataset.",
            )
            knn_run = st.form_submit_button("Run KNN")
    elif st.session_state.stats_plot_select == "Support Vector Machine(SVM) Classification":
        with st.sidebar.form("svm_classification_form"):
            st.selectbox(
                label="Kernel",
                options=("RBF", "Linear", "Polynomial", "Sigmoid"),
                index=0,
                key="svm_kernel",
                help="Kernel transformation used to define the separating boundary. Default: RBF.",
            )
            st.number_input(
                label="C",
                min_value=0.01,
                max_value=100.0,
                step=0.1,
                value=1.0,
                key="svm_C",
                help="Regularization strength; larger values penalize training errors more strongly. Default: 1.0.",
            )
            st.selectbox(
                label="Class weight",
                options=("None", "Balanced"),
                index=0,
                key="svm_class_weight",
                help="Class weighting strategy for imbalanced labels. Default: None.",
            )
            st.number_input(
                label="Polynomial degree",
                min_value=1,
                max_value=10,
                step=1,
                value=3,
                key="svm_degree",
                help="Polynomial kernel degree, used only when Kernel is Polynomial. Default: 3.",
            )
            st.selectbox(
                label="Gamma",
                options=("scale", "auto"),
                index=0,
                key="svm_gamma",
                help="Kernel coefficient for RBF, Polynomial, and Sigmoid kernels. Default: scale.",
            )
            st.slider(
                label="Test size (%)",
                min_value=0,
                max_value=80,
                step=1,
                value=0,
                key="svm_test_size",
                help="Percentage of spectra held out for evaluation. Default: 0%, which fits and evaluates on the full selected dataset.",
            )
            svm_run = st.form_submit_button("Run SVM")
    
    elif st.session_state.stats_plot_select == "Full Spectrum Fitting":
        st.sidebar.info("Note: Your data should be baseline-removed when performing peak fitting.")
        st.sidebar.selectbox(
            label="Algorithm Version",
            options=(
                "Discrete",
                "Prominence-Based"
            ),
            help="Select the fitting algorithm version you would like to use. The Discrete implementation allows you to specify an exact number of components, while the Prominence-Based implementation uses a prominence threshold to determine which peaks to include in the fit.",
            key="fsf_algorithm_version"
        )
        with st.sidebar.form("fsf_form"):

            spectrum_select_options = ["Average"] + [
                column for column in st.session_state.temp.columns
                if column not in ["Ramanshift", "Average", "Standard Deviation"]
            ]
            st.selectbox(
                label="Select Spectrum for Fit",
                options=spectrum_select_options,
                key="fsf_spectrum_select"
            )

            if st.session_state.fsf_algorithm_version == "Discrete":
                st.number_input(
                    label="Number of Peaks (Components) to Fit",
                    value=30,
                    min_value=1,
                    max_value=60,
                    step=1,
                    help="The number of peaks (components) to fit to your data. Expect slower runtimes with greater numbers.",
                    key="fsf_num_peaks"
                )
            if st.session_state.fsf_algorithm_version == "Prominence-Based":
                st.number_input(
                    label="Peak Prominence Threshold",
                    value=100.0,
                    min_value=50.0,
                    max_value=1000.0,
                    step=1.0,
                    help="The minimum prominence required for peaks included in the fit.",
                    key="fsf_prominence_threshold"
                )
            
            st.selectbox(
                label="Peak Shape",
                options=(
                    "Gaussian",
                    "Lorentzian",
                    "Pseudovoigt",
                ),
                help="The mathematical definition of the component curves.",
                key="fsf_peak_shape"
            )

            if st.session_state.fsf_algorithm_version == "Discrete":
                st.number_input(
                    label="Cofit Range Multiplier",
                    value=0.7,
                    min_value=0.4,
                    max_value=1.0,
                    step=0.01,
                    format="%0.2f",
                    help="The distance, relative to the width of any given peak, at which other peaks must be cofit together with this peak. Higher values may improve fit quality with a sacrifice in performance.",
                    key="fsf_cofit_range_multiplier"
                )

                st.selectbox(
                    label="Runtime Control Option",
                    options=(
                        "Quick (m=6)",
                        "Standard (m=9)",
                        "Slow (m=12)",
                        "Thorough (m=15)"
                    ),
                    index=1,
                    help="Controls the worst case runtime by limiting the number of peaks `m` which may be cofit together. Quicker runtimes may result in reduced fit quality.",
                    key="fsf_runtime_control"
                )

            fsf_run = st.form_submit_button("Run Fit")


# Stats section layout
""""""""""""
# Main Page
st.markdown("## Analytics")

if 'df' not in st.session_state:
    st.error('Please go back to Data Upload and upload your data.')

else:
    # st.success("Select a Statistics Plot you wish to see.")
    # if st.session_state.stats_plot:
    
    if 'temp' not in st.session_state:
        st.error('Please process your data, and select Spectra you would like to use.')
    else:
        if st.session_state.get("custom_axis_titles_act", False):
            analytics_x_axis_title = st.session_state.get("custom_x_axis_title", DEFAULT_X_AXIS_TITLE)
            analytics_y_axis_title = st.session_state.get("custom_y_axis_title", DEFAULT_Y_AXIS_TITLE)
        else:
            analytics_x_axis_title = DEFAULT_X_AXIS_TITLE
            analytics_y_axis_title = DEFAULT_Y_AXIS_TITLE

        st.session_state.df_stats = st.session_state.temp.copy()
        st.session_state.df_stats['Average'] = st.session_state.df_stats.iloc[:, 1:].mean(axis=1)
        
        # st.write(st.session_state.df_stats)
        
        stats_data_melted = st.session_state.df_stats.melt(id_vars=['Ramanshift'], var_name='Sample ID', value_name='Intensity')
        
        # st.write(stats_data_melted)
        
        if st.session_state.stats_plot_select == "Average Plot with Original Spectra":
            average_plot_key_suffix = f"{st.session_state.stats_avg_act}_{st.session_state.stats_avg_std_act}"

            if st.session_state.stats_avg_act:
                avg_stats_base = alt.Chart(stats_data_melted).mark_line().encode(
                        x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                        y=alt.Y('Intensity', title=analytics_y_axis_title, type='quantitative'),
                        tooltip=alt.value(None),
                        color=alt.condition(
                            alt.datum['Sample ID'] == 'Average',
                            alt.value('blue'),  # Color for the "Average" sample
                            'Sample ID:N'      # Default color for other samples
                        ),
                        size=alt.condition(
                            alt.datum['Sample ID'] == 'Average',
                            alt.value(3),  # Line width for the "Average" sample
                            alt.value(1)   # Line width for other samples
                        )
                        ).properties(
                            width=1300,
                            height=600,
                            title='Spectra Data Plot'
                        )
                # avg_stats_base = function.style_altair_chart(avg_stats_base)
                # st.altair_chart(avg_stats_base, use_container_width=False)  
                show_plot = avg_stats_base
                log.log_plot_generated_count()
                
            else:
                filtered_avg_df = stats_data_melted[stats_data_melted['Sample ID'] == 'Average']
                avg_stats_base2 = alt.Chart(filtered_avg_df).mark_line().encode(
                        x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                        y=alt.Y('Intensity', title=analytics_y_axis_title, type='quantitative'),
                        tooltip=alt.value(None),
                        color=alt.value('blue'),
                        size=alt.value(3)
                        ).properties(
                            width=1300,
                            height=600,
                            title='Spectra Average Data Plot'
                        )
                
                # st.altair_chart(avg_stats_base2, use_container_width=False)   
                show_plot = avg_stats_base2
                log.log_plot_generated_count()
            
            if st.session_state.stats_avg_std_act:
                ramanshift = st.session_state.df_stats["Ramanshift"]
                # Select only the columns we need for standard deviation calculation
                # columns_to_include = [col for col in st.session_state.temp.columns if col not in ["Ramanshift", "Average"]]
                # df_filtered = st.session_state.temp[columns_to_include]
                # Calculate the standard deviation for the selected columns
                std_df = st.session_state.df_stats.iloc[:, 1:]
                std_df = std_df.drop('Average', axis=1)
                # std_df = std_df.drop('Standard Deviation', axis=1)
                std_values = std_df.std(axis=1)
                
                std_df = pd.DataFrame({
                    "Ramanshift": ramanshift,
                    "Standard Deviation": std_values
                })
                
                # st.write(std_df)
                # Plot the results using Altair
                std_plot = alt.Chart(std_df).mark_line().encode(
                    x=alt.X('Ramanshift', axis=alt.Axis(title=analytics_x_axis_title)),
                    y='Standard Deviation'
                ).properties(
                            width=1300,
                            height=300,
                )
                
                combined_plot = alt.vconcat(show_plot, std_plot).resolve_scale(
                                                x='shared'  # Share the x-axis between the plots
                                            )
                combined_plot = function.style_altair_chart(combined_plot)
                st.altair_chart(
                    combined_plot,
                    use_container_width=False,
                    key=f"analytics_average_plot_combined_{average_plot_key_suffix}"
                )
                log.log_plot_generated_count()
            else:
                show_plot = function.style_altair_chart(show_plot)
                st.altair_chart(
                    show_plot,
                    use_container_width=False,
                    key=f"analytics_average_plot_single_{average_plot_key_suffix}"
                )
            
            stats_download_df = st.session_state.df_stats
            if 'Average' in stats_download_df:
                stats_download_df = stats_download_df.drop('Average', axis=1)

            # Check if 'Standard Deviation' column exists and remove it
            if 'Standard Deviation' in stats_download_df.columns:
                stats_download_df = stats_download_df.drop('Standard Deviation', axis=1)

            # Calculate average and standard deviation for all columns except 'Ramanshift'
            columns_to_calculate = stats_download_df.columns[stats_download_df.columns != 'Ramanshift']
            stats_download_df['Average'] = stats_download_df[columns_to_calculate].mean(axis=1)
            stats_download_df['Standard Deviation'] = stats_download_df[columns_to_calculate].std(axis=1)
            
            @st.cache_data
            def download_df(df):
                return df.to_csv(index = False).encode("utf-8")
            
            # st.write(stats_download_df)
            
            stats_download_df = download_df(stats_download_df)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_file_name = f"data_Average_STD_{current_time}.csv"

            st.download_button(
                label="Download Average and Standard deviation data as CSV",
                data=stats_download_df,
                file_name=download_file_name,
                mime="text/csv",
            )

        elif st.session_state.stats_plot_select == "Confidence Interval Plot":
            
            ramanshift = st.session_state.df_stats["Ramanshift"]
            std_df = st.session_state.df_stats.iloc[:, 1:]
            std_df = std_df.drop('Average', axis=1)
            
            if st.session_state.interval_method == "Confidence Interval":
                threshold = st.session_state.conf_lvl     # 90, 95, 99
                CI_title_text = f"{threshold} percent Confidence Interval Plot"
            else:
                threshold = st.session_state.std_mult     # 1, 2, 3
                CI_title_text = f"±{threshold} Standard Deviation Envelope Plot"

            mean_values, ci_upper, ci_lower = function.confidence_interval(
                df=std_df,
                threshold=threshold,
                interval_method=st.session_state.interval_method
            )

            data = pd.DataFrame({
                'Ramanshift': ramanshift,
                'Mean': mean_values,
                'CI_Upper': ci_upper,
                'CI_Lower': ci_lower
            })

            base = alt.Chart(data).encode(
                x=alt.X('Ramanshift', axis=alt.Axis(title=analytics_x_axis_title))
            ).properties(
                            width=1300,
                            height=600,
                            title = CI_title_text
                )
            
            # Line for mean values
            mean_line = base.mark_line(color='blue').encode(
                y='Mean'
            )

            # Area for confidence interval
            confidence_interval = base.mark_area(color='blue', opacity=0.2).encode(
                y='CI_Lower',
                y2='CI_Upper'
            )

            # Combine the plots
            confidence_plot = confidence_interval + mean_line
            confidence_plot = function.style_altair_chart(confidence_plot)
            st.altair_chart(confidence_plot, use_container_width=False)
            log.log_plot_generated_count()

        elif st.session_state.stats_plot_select == "Spectral Derivation":
            with st.sidebar:
                
                # Window length input
                win = st.number_input(
                    "Window length (odd, ≥ 5)",
                    min_value=3,
                    max_value=25,  # optional safeguard if df is defined
                    step=2,
                    value=11,
                    help="Controls smoothing span. Must be odd and at least 5. "
                        "Larger values = stronger smoothing but risk of oversmoothing peaks."
                )

                # Polynomial order input
                poly = st.number_input(
                    "Polynomial order (< window length)",
                    min_value=2,
                    max_value=7,  # practical upper bound; can raise if needed
                    step=1,
                    value=3,
                    help="Controls local polynomial fitting. "
                        "Order must be smaller than the window length. "
                        "Typical choices: 2–3 for smooth baseline, 4–5 for sharper peaks."
                )

            try:
                # Expecting `stats_data_melted` with columns: 'Ramanshift', 'Intensity', 'Sample ID'
                df = stats_data_melted.copy()

                # Ensure numeric and sorted within each sample
                df["Ramanshift"] = pd.to_numeric(df["Ramanshift"], errors="coerce")
                df["Intensity"]  = pd.to_numeric(df["Intensity"],  errors="coerce")
                df = df.dropna(subset=["Ramanshift", "Intensity"])

                proc = df.groupby("Sample ID", group_keys=False).apply(function.spectra_derivation,    norm_method=st.session_state.deriv_norm_method,  # value from selectbox
                sg_win=win,                  # from number_input
                sg_poly=poly                 # from number_input
            )

                # --- Build Altair charts (two panels: 1st & 2nd derivative) ---
                # highlight "Average" if present
                highlight_cond = alt.datum["Sample ID"] == "Average"

                first_deriv = (
                    alt.Chart(proc)
                    .mark_line()
                    .encode(
                        x=alt.X("Ramanshift:Q", title=analytics_x_axis_title),
                        y=alt.Y("y1:Q", title="1st derivative (a.u./cm⁻¹)"),
                        color=alt.condition(highlight_cond, alt.value("blue"), alt.Color("Sample ID:N", title="Sample")),
                        # size=alt.condition(highlight_cond, alt.value(3), alt.value(1)),
                        tooltip=alt.value(None),
                    )
                    .properties(width=1300, height=300, title="First Derivative")
                )

                second_deriv = (
                    alt.Chart(proc)
                    .mark_line()
                    .encode(
                        x=alt.X("Ramanshift:Q", title=analytics_x_axis_title),
                        y=alt.Y("y2:Q", title="2nd derivative (a.u./cm⁻²)"),
                        color=alt.condition(highlight_cond, alt.value("blue"), alt.Color("Sample ID:N", title="Sample")),
                        # size=alt.condition(highlight_cond, alt.value(3), alt.value(1)),
                        tooltip=alt.value(None),
                    )
                    .properties(width=1300, height=300, title="Second Derivative")
                )

                show_plot = first_deriv & second_deriv  # vertical concat
                show_plot = function.style_altair_chart(show_plot)
                st.altair_chart(show_plot, use_container_width=False)

                # --- Build tidy DataFrames for export ---
                # y1_df = (
                #     proc_plus[["Ramanshift", "Sample ID", "y1"]]
                #     .rename(columns={"y1": "FirstDerivative"})
                #     .sort_values(["Sample ID", "Ramanshift"])
                # )

                # y2_df = (
                #     proc_plus[["Ramanshift", "Sample ID", "y2"]]
                #     .rename(columns={"y2": "SecondDerivative"})
                #     .sort_values(["Sample ID", "Ramanshift"])
                # )

                # # (Optional) If you prefer wide format (each Sample ID = one column), uncomment:
                # # y1_df = y1_df.pivot(index="Ramanshift", columns="Sample ID", values="FirstDerivative").reset_index()
                # # y2_df = y2_df.pivot(index="Ramanshift", columns="Sample ID", values="SecondDerivative").reset_index()

                # @st.cache_data
                # def _to_csv_bytes(df: pd.DataFrame) -> bytes:
                #     # utf-8 without BOM; change to "utf-8-sig" if Excel encoding is needed
                #     return df.to_csv(index=False).encode("utf-8")

                # y1_csv = _to_csv_bytes(y1_df)
                # y2_csv = _to_csv_bytes(y2_df)

                # # --- File names with timestamp & norm method (if available in your state) ---
                # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                # norm_tag = str(st.session_state.get("deriv_norm_method", "None")).replace(" ", "")
                # fname_y1 = f"SERS_FirstDerivative_{norm_tag}_{current_time}.csv"
                # fname_y2 = f"SERS_SecondDerivative_{norm_tag}_{current_time}.csv"

                # # --- Two side-by-side download buttons ---
                # c1, c2 = st.columns(2)
                # with c1:
                #     st.download_button(
                #         label="⬇️ Download 1st Derivative (CSV)",
                #         data=y1_csv,
                #         file_name=fname_y1,
                #         mime="text/csv",
                #     )
                # with c2:
                #     st.download_button(
                #         label="⬇️ Download 2nd Derivative (CSV)",
                #         data=y2_csv,
                #         file_name=fname_y2,
                #         mime="text/csv",
                    # )
                # optional: your logger
                log.log_function_call("Analytics_Spectral_Derivation",
                                        f_params={
                                            'norm_method':st.session_state.deriv_norm_method,
                                            'sg_win':win,
                                            'sg_poly':poly
                                        })
                log.log_plot_generated_count()
            except Exception as e:
                st.error(f"Error during processing: {e}")

        elif st.session_state.stats_plot_select == "Fast Fourier Transform (FFT) analysis":
            selected_fft_target = st.session_state.get("fft_target_spectrum", "Average")
            filtered_fft_df = stats_data_melted[stats_data_melted["Sample ID"] == selected_fft_target]
            try:
                filtered_fft_df = filtered_fft_df.copy()
                filtered_fft_df["Ramanshift"] = pd.to_numeric(filtered_fft_df["Ramanshift"], errors="coerce")
                filtered_fft_df["Intensity"] = pd.to_numeric(filtered_fft_df["Intensity"], errors="coerce")
                filtered_fft_df = filtered_fft_df.dropna(subset=["Ramanshift", "Intensity"])

                if filtered_fft_df.empty:
                    raise ValueError(f"No valid data found for spectrum '{selected_fft_target}'.")

                fft_plot_title = f"FFT: {selected_fft_target}"
                frequency_axis_title = "Positive Frequency (cycles/cm^-1)"
                fft_df = function.compute_fft_spectrum(
                    ramanshift=filtered_fft_df["Ramanshift"].to_numpy(),
                    intensity=filtered_fft_df["Intensity"].to_numpy(),
                    source_spectrum=selected_fft_target,
                    subtract_average=st.session_state.get("fft_subtract_average", False)
                )
                fft_plots = function.build_fft_plots(
                    fft_df=fft_df,
                    frequency_axis_title=frequency_axis_title,
                    phase_axis_title="Phase (deg)",
                    amplitude_axis_title="Amplitude",
                    real_axis_title="Real",
                    imaginary_axis_title="Imaginary",
                    power_axis_title="Power (MSA)",
                    title_prefix=fft_plot_title
                )

                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.altair_chart(fft_plots["amplitude"], use_container_width=True)
                with row1_col2:
                    st.altair_chart(fft_plots["phase"], use_container_width=True)

                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.altair_chart(fft_plots["power"], use_container_width=True)
                with row2_col2:
                    st.altair_chart(fft_plots["real_imaginary"], use_container_width=True)

                row3_col1, row3_col2 = st.columns(2)
                with row3_col1:
                    st.altair_chart(fft_plots["real"], use_container_width=True)
                with row3_col2:
                    st.altair_chart(fft_plots["imaginary"], use_container_width=True)

                for _ in range(6):
                    log.log_plot_generated_count()
                log.log_function_call(
                    "Analytics_FFT",
                    f_params={
                        "target_spectrum": selected_fft_target,
                        "subtract_average": st.session_state.get("fft_subtract_average", False)
                    }
                )

                @st.cache_data
                def download_fft_df(df):
                    return df.to_csv(index=False).encode("utf-8")

                fft_download_df = download_fft_df(fft_df)
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                download_file_name = f"data_FFT_{selected_fft_target}_{current_time}.csv"

                st.download_button(
                    label="Download FFT analysis data as CSV",
                    data=fft_download_df,
                    file_name=download_file_name,
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during FFT processing: {e}")

        elif st.session_state.stats_plot_select == "Correlation Heatmap":
            # Select only the columns we need for correlation calculation
            # Filter out the columns
            columns_to_include = [col for col in st.session_state.temp.columns if col not in ["Ramanshift", "Average","Standard Deviation"]]
            df_filtered = st.session_state.temp[columns_to_include].copy()
            
            # Add average column if toggle is enabled
            if st.session_state.heatmap_compute_avg:
                df_filtered['Average'] = df_filtered.mean(axis=1)
            
            # Calculate the correlation matrix based on selected method
            if st.session_state.heatmap_corr_method == "Pearson Correlation":
                corr_matrix = df_filtered.corr(method='pearson')
            else:  # Cosine Similarity
                from sklearn.metrics.pairwise import cosine_similarity
                # Calculate cosine similarity between columns (transpose so columns become rows)
                cosine_sim = cosine_similarity(df_filtered.T)
                corr_matrix = pd.DataFrame(
                    cosine_sim,
                    index=df_filtered.columns,
                    columns=df_filtered.columns
                )

            # Display the correlation matrix
            # stats_row2.dataframe(corr_matrix, use_container_width=True)

            # Melt the correlation matrix for Altair
            corr_melted = corr_matrix.reset_index().melt(id_vars='index', var_name='Variable', value_name='Correlation')
            
            display_text = len(columns_to_include) <= 10
            
            if st.session_state.heatmap_scale:
                # heatmap = alt.Chart(corr_melted).mark_rect().encode(
                #     x=alt.X('Variable', title=None, sort=None),
                #     y=alt.Y('index', title=None, sort=None),
                #     color=alt.Color('Correlation', scale=alt.Scale(domain=[st.session_state.heatmap_min, st.session_state.heatmap_max], scheme='yelloworangered')),
                #     tooltip=[alt.Tooltip('Variable', title='X Axis'), 
                #             alt.Tooltip('index', title='Y Axis'), 
                #             alt.Tooltip('Correlation', title='Correlation')]
                # ).properties(
                #     width=800,
                #     height=800,
                #     title='Correlation Matrix Heatmap'
                # ).configure_title(
                #     anchor='start'
                # )
                # Base heatmap
                base = alt.Chart(corr_melted).encode(
                    x=alt.X('Variable', title=None, sort=None),
                    y=alt.Y('index', title=None, sort=None)
                )

                # Heatmap layer
                heatmap = base.mark_rect().encode(
                    color=alt.Color('Correlation', scale=alt.Scale(domain=[st.session_state.heatmap_min, st.session_state.heatmap_max], scheme='yelloworangered')),
                    tooltip=[
                        alt.Tooltip('Variable', title='X'), 
                        alt.Tooltip('index', title='Y'), 
                        alt.Tooltip('Correlation', title='Correlation')
                    ]
                )

                # Conditional text layer for correlation values
                if display_text:
                    text = base.mark_text(baseline='middle', size=15).encode(  # Adjust size value as needed
                        text=alt.Text('Correlation:Q', format='.3f'),
                        color=alt.condition(
                            alt.datum.Correlation > 0.5, 
                            alt.value('black'),  # Change color for better visibility if needed
                            alt.value('white')
                        )
                    )
                    combined = alt.layer(heatmap, text).properties(
                        width=800,
                        height=800,
                        title='Correlation Matrix Heatmap'
                    ).configure_title(
                        anchor='start'
                    )
                else:
                    combined = heatmap.properties(
                        width=800,
                        height=800,
                        title='Correlation Matrix Heatmap'
                    ).configure_title(
                        anchor='start'
                    )
                
            else:
                # heatmap = alt.Chart(corr_melted).mark_rect().encode(
                #     x=alt.X('Variable', title=None, sort=None),
                #     y=alt.Y('index', title=None, sort=None),
                #     color=alt.Color('Correlation', scale=alt.Scale(scheme='yelloworangered')),
                #     tooltip=[alt.Tooltip('Variable', title='X'), 
                #             alt.Tooltip('index', title='Y'), 
                #             alt.Tooltip('Correlation', title='Correlation')]
                # ).properties(
                #     width=800,
                #     height=800,
                #     title='Correlation Matrix Heatmap'
                # ).configure_title(
                #     anchor='start'
                # )
                
                # Base heatmap
                base = alt.Chart(corr_melted).encode(
                    x=alt.X('Variable', title=None, sort=None),
                    y=alt.Y('index', title=None, sort=None)
                )

                # Heatmap layer
                heatmap = base.mark_rect().encode(
                    color=alt.Color('Correlation', scale=alt.Scale(scheme='yelloworangered')),
                    tooltip=[
                        alt.Tooltip('Variable', title='X'), 
                        alt.Tooltip('index', title='Y'), 
                        alt.Tooltip('Correlation', title='Correlation')
                    ]
                )

                # Conditional text layer for correlation values
                if display_text:
                    text = base.mark_text(baseline='middle', size=15).encode(  # Adjust size value as needed
                        text=alt.Text('Correlation:Q', format='.3f'),
                        color=alt.condition(
                            alt.datum.Correlation > 0.5, 
                            alt.value('black'),  # Change color for better visibility if needed
                            alt.value('white')
                        )
                    )
                    combined = alt.layer(heatmap, text).properties(
                        width=800,
                        height=800,
                        title='Correlation Matrix Heatmap'
                    ).configure_title(
                        anchor='start'
                    )
                else:
                    combined = heatmap.properties(
                        width=800,
                        height=800,
                        title='Correlation Matrix Heatmap'
                    ).configure_title(
                        anchor='start'
                    )

            # Display the heatmap
            combined = function.style_altair_chart(combined)
            st.altair_chart(combined, use_container_width=False)
            log.log_plot_generated_count()
            log.log_function_call("Analytics_Correlation_Heatmap",
                                    f_params={
                                        'method':st.session_state.heatmap_corr_method,
                                        'compute_avg':st.session_state.heatmap_compute_avg
                                    })
            
            @st.cache_data
            def download_df(df):
                return df.to_csv(index = True).encode("utf-8")
            
            # st.write(stats_download_df)
            
            stats_download_df = download_df(corr_matrix)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_file_name = f"data_Corr_matrix_{current_time}.csv"

            st.download_button(
                label="Download Correlation Matrix as CSV",
                data=stats_download_df,
                file_name=download_file_name,
                mime="text/csv",
            )

# """
#                 elif std_option == "Confidence Interval Plot":
                    # # 3rd plot
                    # mean_values = df_filtered.mean(axis=1)
                    # std_values = df_filtered.std(axis=1)
                    
                    # ci_upper = mean_values + std_values
                    # ci_lower = mean_values - std_values
                    
                    # # Create the plot
                    # fig1, ax1 = plt.subplots(figsize=(10, 6))
                    # ax1.plot(ramanshift, mean_values, label='Mean', color='blue')
                    # ax1.fill_between(ramanshift, ci_lower, ci_upper, color='blue', alpha=0.2, label='Confidence Interval')

                    # # Add labels, title, and legend
                    # ax1.set_xlabel('Ramanshift')
                    # ax1.set_ylabel('Value')
                    # ax1.set_title('Confidence Interval Plot')
                    # ax1.legend()
                    # ax1.grid(True)
                    
                    # stats_row.write(" ")
                    # stats_row.pyplot(fig1, use_container_width= True)


#             # 3nd row std
#             stats_row2 = row([0.3, 0.7])
#             stats_row2.toggle('Calculate Correlation for Spectra selected', key = 'corr_act')
#             # try: 
#             if st.session_state.corr_act:
#                 # Select only the columns we need for standard deviation calculation
#                 # Filter out the columns
#                 columns_to_include = [col for col in st.session_state.temp.columns if col not in ["Ramanshift", "Average"]]
#                 df_filtered = st.session_state.temp[columns_to_include]

#                 # Calculate the correlation matrix
#                 corr_matrix = df_filtered.corr()

#                 # Display the correlation matrix
#                 # stats_row2.dataframe(corr_matrix, use_container_width=True)

#                 # Melt the correlation matrix for Altair
#                 corr_melted = corr_matrix.reset_index().melt(id_vars='index', var_name='Variable', value_name='Correlation')

#                 # Generate the heatmap with Altair
#                 heatmap = alt.Chart(corr_melted).mark_rect().encode(
#                     x=alt.X('Variable', title=None, sort=None),
#                     y=alt.Y('index', title=None, sort=None),
#                     color=alt.Color('Correlation', scale=alt.Scale(scheme='yellowgreenblue')),
#                     tooltip=['Correlation']
#                 ).properties(
#                     width=600,
#                     height=600,
#                     title='Correlation Matrix Heatmap'
#                 ).configure_title(
#                     anchor='start'
#                 )

#                 # Display the heatmap
#                 stats_row2.altair_chart(heatmap, use_container_width=False)
# """
        elif st.session_state.stats_plot_select == "Peak Identification and Stats":   
            selected_peak_target = st.session_state.get("peak_identification_target", "Average")
            filtered_peak_df = stats_data_melted[stats_data_melted['Sample ID'] == selected_peak_target]
            
            # avg_stats_base2 = alt.Chart(filtered_avg_df).mark_line().encode(
            #         x=alt.X('Ramanshift', title='Raman shift/cm^-1', type='quantitative'),
            #         y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
            #         tooltip=alt.value(None),
            #         color=alt.value('blue'),
            #         size=alt.value(3)
            #         ).properties(
            #             width=1300,
            #             height=600,
            #             title='Spectra Average Data Plot'
            #         )
                
            # # st.altair_chart(avg_stats_base2, use_container_width=False)   
            # show_plot = avg_stats_base2 
            # st.altair_chart(show_plot)
            # st.write(filtered_avg_df)
            
            peaks, properties = function.peak_identification(spectra=filtered_peak_df['Intensity'].to_numpy(),
                                                            height= st.session_state.peak_iden_height_p,
                                                            threshold = st.session_state.peak_iden_threshold_p,
                                                            distance = st.session_state.peak_iden_distance_p,
                                                            prominence = st.session_state.peak_iden_prominence_p,
                                                            width = st.session_state.peak_iden_width_p)


            # Extract Raman shift and intensity values
            raman_shift = filtered_peak_df['Ramanshift']
            intensity = filtered_peak_df['Intensity']

            # Plot the Raman shift vs. Intensity with detected peaks using .iloc
            # fig = plt.figure(figsize=(10, 6))
            # plt.plot(raman_shift, intensity, label='Intensity')
            # plt.plot(raman_shift.iloc[peaks], intensity.iloc[peaks], "ro", markersize=8, label='Peaks')  # Red 'o' markers for peaks
            # plt.title('Peak Detection in Raman Shift Data Using peak_identification Function')
            # plt.xlabel('Raman Shift')
            # plt.ylabel('Intensity')
            # plt.legend()
            # st.pyplot(fig)
            
            
            # peaks, _ = peak_identification(spectra=filtered_avg_df['Intensity'].to_numpy(), prominence=1.0)

            # Step 2: Prepare a DataFrame for the peak markers
            peak_df = filtered_peak_df.iloc[peaks].copy()
            
            properties_df = pd.DataFrame(properties)
            
            # properties_df['peak_heights'] = [peaks].values
            
            # st.write(properties_df)
            
            peak_df = peak_df.reset_index(drop=True)
            properties_df = properties_df.reset_index(drop=True)
            
            # st.write(peak_df)
            
            peak_df = pd.concat([peak_df, properties_df], axis=1)
            
            # st.write(peak_df)
            
            if st.session_state.peak_iden_auto:
                peak_df = peak_df.sort_values(by='prominences', ascending=False).head(st.session_state.peak_iden_auto_num)
            else:
                peak_df = peak_df.sort_values(by='Intensity', ascending=False)

            
            # Step 3: Create the base interactive plot
            peak_plot_title = f'Peak Identification: {selected_peak_target}'

            avg_stats_base2 = alt.Chart(filtered_peak_df).mark_line().encode(
                x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                y=alt.Y('Intensity', title=analytics_y_axis_title, type='quantitative'),
                tooltip=alt.value(None),
                color=alt.value('blue'),
                size=alt.value(3)
            ).properties(
                width=1300,
                height=600,
                title=peak_plot_title
            )

            # Step 4: Create the peak markers plot
            peak_markers = alt.Chart(peak_df).mark_point(
                filled=True,
                color='red',
                size=100
            ).encode(
                x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                y=alt.Y('Intensity', title=analytics_y_axis_title, type='quantitative'),
                tooltip=[alt.Tooltip('Ramanshift', title=analytics_x_axis_title),
                        alt.Tooltip('Intensity', title=analytics_y_axis_title)]
            ).properties(
                width=1300,
                height=600,
                title=peak_plot_title
            )

            # Step 5: Combine the base plot and peak markers
            interactive_plot = (avg_stats_base2 + peak_markers).properties(
                width=1300,
                height=600
            ).interactive()
            interactive_plot = function.style_altair_chart(interactive_plot)

            # Display the plot in Streamlit (if using Streamlit)
            st.altair_chart(interactive_plot, use_container_width=False)
            
            
            st.write("**Peak property**")
            
            st.write(peak_df)
            
            log.log_plot_generated_count()
            log.log_function_call("Analytics_Peak_Identification",
                                    f_params={
                                        'target_spectrum': selected_peak_target,
                                        'auto':st.session_state.peak_iden_auto,
                                        'height':st.session_state.peak_iden_height_p,
                                        'threshold':st.session_state.peak_iden_threshold_p,
                                        'distance':st.session_state.peak_iden_distance_p,
                                        'prominence':st.session_state.peak_iden_prominence_p,
                                        'width':st.session_state.peak_iden_width_p
                                    })
        
        elif st.session_state.stats_plot_select == "Hierarchically-clustered Heatmap":
            
            st.write("**Hierarchically-clustered Heatmap**")
            
            temp = st.session_state.temp.drop(columns=['Average'], errors='ignore')
            
            if st.session_state.HCA_heatmap:
                st.pyplot(function.hierarchical_clustering_heatmap(temp))
                log.log_function_call("Analytics_Clustering_Clustermap", f_params={})
            else:
                st.pyplot(function.hierarchical_clustering_tree(temp))
                log.log_function_call("Analytics_Clustering_Dendrogram", f_params={})
        
            log.log_plot_generated_count()
        
        elif st.session_state.stats_plot_select == "Principal Components Analysis (PCA)":
            
            temp = st.session_state.temp.drop(columns=['Average'], errors='ignore')
            label_df = st.session_state.get('label_df')

            if label_df is None:
                st.warning(
                    "No label table found in session. "
                    "Proceeding with default label = 1 for every spectrum."
                )
                label_df = pd.DataFrame({
                    'Spectrum': temp.columns[1:],   # skip RamanShift column
                    'Label':    1,
                    'Note':     ' '
                })

            # ------------------------------------------------------------------
            # Ensure the first column is named exactly 'Ramanshift'
            # ------------------------------------------------------------------
            first_col = label_df.columns[0]
            if first_col != 'Ramanshift':
                label_df = label_df.rename(columns={first_col: 'Ramanshift'})
            # ------------------------------------------------------------------
            # 2.  Run PCA (function.pca expects label_df and a flag)
            # ------------------------------------------------------------------
            pca_result_df, pc1_vs_pc2_plot, cumulative_variance_plot, loading_plot = (
                function.pca(
                    temp,
                    is_label=True,                 # always True now – we supply label_df
                    label_df=label_df,
                    horizontal_pc=st.session_state.PCA_horizontal,
                    vertical_pc=st.session_state.PCA_vertical
                )
            )
            # Re‑order columns for display
            desired_first = ['Ramanshift', 'Label']
            pc_cols = [c for c in pca_result_df.columns if c.upper().startswith('PC')]
            other_cols = [c for c in pca_result_df.columns
                        if c not in desired_first + pc_cols]

            new_order = [c for c in desired_first if c in pca_result_df.columns] + pc_cols + other_cols
            pca_result_df = pca_result_df[new_order]

            # ------------------------------------------------------------------
            # 3.  Display results
            # ------------------------------------------------------------------
            st.altair_chart(function.style_altair_chart(pc1_vs_pc2_plot), use_container_width=False)
            log.log_plot_generated_count()

            st.altair_chart(function.style_altair_chart(cumulative_variance_plot), use_container_width=False)
            log.log_plot_generated_count()

            st.altair_chart(function.style_altair_chart(loading_plot), use_container_width=False)
            log.log_plot_generated_count()

            log.log_function_call("Analytics_PCA", f_params={})

            st.write("### PCA scores table")
            st.write(pca_result_df)

        elif st.session_state.stats_plot_select == "T-SNE Dimensionality Reduction":

            st.write("**T‑Distributed stochastic neighbor embedding (t‑SNE)**")

            temp = st.session_state.temp.drop(columns=['Average'], errors='ignore')

            label_df = st.session_state.get('label_df')   # could be None

            tsne_df, tsne_plot = function.tsne(
                temp,
                perplexity=st.session_state.tSNE_perplexity,
                n_iter=st.session_state.tSNE_n_iter,
                label_df=label_df
            )
            st.altair_chart(function.style_altair_chart(tsne_plot), use_container_width=False)

            log.log_plot_generated_count()
            log.log_function_call("Analytics_TSNE",
                                    f_params={
                                        'perplexity':st.session_state.tSNE_perplexity,
                                        'n_iter':st.session_state.tSNE_n_iter,
                                    })

            st.write(tsne_df)

        elif st.session_state.stats_plot_select == "Random Forest(RF) Classification":
            st.write("**Random Forest(RF) Classification**")
            if not rf_run:
                st.info("Set Random Forest parameters in the sidebar, then click Run Random Forest.")
            elif st.session_state.get("label_df") is None:
                st.error("Classification requires label data. Upload or assign labels before running this analysis.")
            else:
                try:
                    temp = st.session_state.temp.drop(columns=["Average"], errors="ignore")
                    rf_result = function.analytics_ml_classification_random_forest(
                        temp,
                        st.session_state.label_df,
                        n_estimators=st.session_state.rf_n_estimators,
                        max_depth=st.session_state.rf_max_depth,
                        min_samples_leaf=st.session_state.rf_min_samples_leaf,
                        test_size=st.session_state.rf_test_size,
                    )

                    split_info = rf_result.get("split_info", {})
                    if split_info.get("info_message"):
                        st.info(split_info["info_message"])

                    eda = rf_result.get("eda", {})
                    if eda:
                        st.write("### Class Counts by Split")
                        st.caption("Tabulates the fitted split composition by class; percentages are normalized within each split.")
                        class_counts_df = eda["class_counts"].copy()
                        for percent_column in [col for col in class_counts_df.columns if "Percent" in col]:
                            class_counts_df[percent_column] = class_counts_df[percent_column].map(lambda value: f"{value:.1%}")
                        st.dataframe(class_counts_df, use_container_width=True)

                        st.write("### Class Balance")
                        st.caption("Compares class-frequency distributions across the exact samples used for model fitting and evaluation.")
                        st.altair_chart(eda["class_count_chart"], use_container_width=False)
                        log.log_plot_generated_count()

                        spectra_envelopes = eda.get("spectra_envelopes", [])
                        if rf_result.get("mode") == "train_test_split" and len(spectra_envelopes) == 2:
                            st.write("### Split Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            train_eda_col, test_eda_col = st.columns(2)
                            with train_eda_col:
                                st.altair_chart(spectra_envelopes[0]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                            with test_eda_col:
                                st.altair_chart(spectra_envelopes[1]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                        else:
                            st.write("### Full Dataset Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            for envelope in spectra_envelopes:
                                st.altair_chart(envelope["chart"], use_container_width=False)
                                log.log_plot_generated_count()

                    if rf_result.get("mode") == "train_test_split" and len(rf_result["sections"]) == 2:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination for each fitted split.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            section = rf_result["sections"][0]
                            st.write("#### Training Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                        with test_col:
                            section = rf_result["sections"][1]
                            st.write("#### Test Performance")
                            st.dataframe(section["metrics"], use_container_width=False)

                        st.write("### Confusion Matrix")
                        st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(rf_result["sections"][0]["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()

                        st.write("### ROC Curve")
                        st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(rf_result["sections"][0]["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                    else:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination on the full selected dataset.")
                        for section in rf_result["sections"]:
                            st.write("#### Full Dataset Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                            st.write("### Full Dataset Confusion Matrix")
                            st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                            st.write("### Full Dataset ROC Curve")
                            st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()

                    # Model-specific extra plots are intentionally hidden for now.
                    # for extra_plot in rf_result.get("extra_plots", []):
                    #     st.write(f"### {extra_plot['name']}")
                    #     st.altair_chart(extra_plot["chart"], use_container_width=False)
                    #     log.log_plot_generated_count()

                    log.log_function_call(
                        "Analytics_ML_Classification_Random_Forest",
                        f_params={
                            "n_estimators": st.session_state.rf_n_estimators,
                            "max_depth": st.session_state.rf_max_depth,
                            "min_samples_leaf": st.session_state.rf_min_samples_leaf,
                            "test_size": st.session_state.rf_test_size,
                        },
                    )
                except Exception as e:
                    st.error(f"Error running Random Forest classification: {e}")

        elif st.session_state.stats_plot_select == "K-Nearest Neighbors(KNN) Classification":
            st.write("**K-Nearest Neighbors(KNN) Classification**")
            if not knn_run:
                st.info("Set KNN parameters in the sidebar, then click Run KNN.")
            elif st.session_state.get("label_df") is None:
                st.error("Classification requires label data. Upload or assign labels before running this analysis.")
            else:
                try:
                    temp = st.session_state.temp.drop(columns=["Average"], errors="ignore")
                    knn_result = function.analytics_ml_classification_knn(
                        temp,
                        st.session_state.label_df,
                        n_neighbors=st.session_state.knn_n_neighbors,
                        test_size=st.session_state.knn_test_size,
                        weights=st.session_state.knn_weights,
                        metric=st.session_state.knn_metric,
                    )

                    split_info = knn_result.get("split_info", {})
                    if split_info.get("info_message"):
                        st.info(split_info["info_message"])

                    eda = knn_result.get("eda", {})
                    if eda:
                        st.write("### Class Counts by Split")
                        st.caption("Tabulates the fitted split composition by class; percentages are normalized within each split.")
                        class_counts_df = eda["class_counts"].copy()
                        for percent_column in [col for col in class_counts_df.columns if "Percent" in col]:
                            class_counts_df[percent_column] = class_counts_df[percent_column].map(lambda value: f"{value:.1%}")
                        st.dataframe(class_counts_df, use_container_width=True)

                        st.write("### Class Balance")
                        st.caption("Compares class-frequency distributions across the exact samples used for model fitting and evaluation.")
                        st.altair_chart(eda["class_count_chart"], use_container_width=False)
                        log.log_plot_generated_count()

                        spectra_envelopes = eda.get("spectra_envelopes", [])
                        if knn_result.get("mode") == "train_test_split" and len(spectra_envelopes) == 2:
                            st.write("### Split Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            train_eda_col, test_eda_col = st.columns(2)
                            with train_eda_col:
                                st.altair_chart(spectra_envelopes[0]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                            with test_eda_col:
                                st.altair_chart(spectra_envelopes[1]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                        else:
                            st.write("### Full Dataset Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            for envelope in spectra_envelopes:
                                st.altair_chart(envelope["chart"], use_container_width=False)
                                log.log_plot_generated_count()

                    if knn_result.get("mode") == "train_test_split" and len(knn_result["sections"]) == 2:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination for each fitted split.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            section = knn_result["sections"][0]
                            st.write("#### Training Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                        with test_col:
                            section = knn_result["sections"][1]
                            st.write("#### Test Performance")
                            st.dataframe(section["metrics"], use_container_width=False)

                        st.write("### Confusion Matrix")
                        st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(knn_result["sections"][0]["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()

                        st.write("### ROC Curve")
                        st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(knn_result["sections"][0]["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                    else:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination on the full selected dataset.")
                        for section in knn_result["sections"]:
                            st.write("#### Full Dataset Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                            st.write("### Full Dataset Confusion Matrix")
                            st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                            st.write("### Full Dataset ROC Curve")
                            st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()

                    log.log_function_call(
                        "Analytics_ML_Classification_KNN",
                        f_params={
                            "n_neighbors": st.session_state.knn_n_neighbors,
                            "weights": st.session_state.knn_weights,
                            "metric": st.session_state.knn_metric,
                            "test_size": st.session_state.knn_test_size,
                        },
                    )
                except Exception as e:
                    st.error(f"Error running KNN classification: {e}")

        elif st.session_state.stats_plot_select == "Support Vector Machine(SVM) Classification":
            st.write("**Support Vector Machine(SVM) Classification**")
            if not svm_run:
                st.info("Set SVM parameters in the sidebar, then click Run SVM.")
            elif st.session_state.get("label_df") is None:
                st.error("Classification requires label data. Upload or assign labels before running this analysis.")
            else:
                try:
                    temp = st.session_state.temp.drop(columns=["Average"], errors="ignore")
                    svm_result = function.analytics_ml_classification_svm(
                        temp,
                        st.session_state.label_df,
                        test_size=st.session_state.svm_test_size,
                        kernel=st.session_state.svm_kernel,
                        C=st.session_state.svm_C,
                        class_weight=st.session_state.svm_class_weight,
                        degree=st.session_state.svm_degree,
                        gamma=st.session_state.svm_gamma,
                    )

                    split_info = svm_result.get("split_info", {})
                    if split_info.get("info_message"):
                        st.info(split_info["info_message"])

                    eda = svm_result.get("eda", {})
                    if eda:
                        st.write("### Class Counts by Split")
                        st.caption("Tabulates the fitted split composition by class; percentages are normalized within each split.")
                        class_counts_df = eda["class_counts"].copy()
                        for percent_column in [col for col in class_counts_df.columns if "Percent" in col]:
                            class_counts_df[percent_column] = class_counts_df[percent_column].map(lambda value: f"{value:.1%}")
                        st.dataframe(class_counts_df, use_container_width=True)

                        st.write("### Class Balance")
                        st.caption("Compares class-frequency distributions across the exact samples used for model fitting and evaluation.")
                        st.altair_chart(eda["class_count_chart"], use_container_width=False)
                        log.log_plot_generated_count()

                        spectra_envelopes = eda.get("spectra_envelopes", [])
                        if svm_result.get("mode") == "train_test_split" and len(spectra_envelopes) == 2:
                            st.write("### Split Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            train_eda_col, test_eda_col = st.columns(2)
                            with train_eda_col:
                                st.altair_chart(spectra_envelopes[0]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                            with test_eda_col:
                                st.altair_chart(spectra_envelopes[1]["chart"], use_container_width=False)
                                log.log_plot_generated_count()
                        else:
                            st.write("### Full Dataset Class Spectra Visualization")
                            st.caption("Class mean spectra are shown as lines; shaded envelopes span the class-wise minimum-to-maximum intensity range.")
                            for envelope in spectra_envelopes:
                                st.altair_chart(envelope["chart"], use_container_width=False)
                                log.log_plot_generated_count()

                    if svm_result.get("mode") == "train_test_split" and len(svm_result["sections"]) == 2:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination for each fitted split.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            section = svm_result["sections"][0]
                            st.write("#### Training Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                        with test_col:
                            section = svm_result["sections"][1]
                            st.write("#### Test Performance")
                            st.dataframe(section["metrics"], use_container_width=False)

                        st.write("### Confusion Matrix")
                        st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(svm_result["sections"][0]["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()

                        st.write("### ROC Curve")
                        st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                        train_col, test_col = st.columns(2)
                        with train_col:
                            st.altair_chart(svm_result["sections"][0]["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                        with test_col:
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()
                    else:
                        st.write("### Performance Table")
                        st.caption("Macro-averaged metrics summarize model discrimination on the full selected dataset.")
                        for section in svm_result["sections"]:
                            st.write("#### Full Dataset Performance")
                            st.dataframe(section["metrics"], use_container_width=False)
                            st.write("### Full Dataset Confusion Matrix")
                            st.caption("Rows encode observed classes and columns encode predicted classes; diagonal counts are correct predictions.")
                            st.altair_chart(section["confusion_matrix"], use_container_width=False)
                            log.log_plot_generated_count()
                            st.write("### Full Dataset ROC Curve")
                            st.caption("One-vs-rest ROC curves quantify class separability; larger AUC values indicate stronger discrimination.")
                            st.altair_chart(section["roc_curve"], use_container_width=False)
                            log.log_plot_generated_count()

                    # Model-specific extra plots are intentionally hidden for now.
                    # for extra_plot in svm_result.get("extra_plots", []):
                    #     st.write(f"### {extra_plot['name']}")
                    #     st.altair_chart(extra_plot["chart"], use_container_width=False)
                    #     log.log_plot_generated_count()

                    log.log_function_call(
                        "Analytics_ML_Classification_SVM",
                        f_params={
                            "kernel": st.session_state.svm_kernel,
                            "C": st.session_state.svm_C,
                            "class_weight": st.session_state.svm_class_weight,
                            "degree": st.session_state.svm_degree,
                            "gamma": st.session_state.svm_gamma,
                            "test_size": st.session_state.svm_test_size,
                        },
                    )
                except Exception as e:
                    st.error(f"Error running SVM classification: {e}")

        elif st.session_state.stats_plot_select == "Full Spectrum Fitting":
            import numpy as np
            import time
            st.write("**Full Spectrum Fitting**")
            if not fsf_run and 'fsf_results_df' not in st.session_state:
                st.info("Set up fit parameters, then click 'Run Fit.'")
            else:
                if fsf_run:
                    filtered_fsf_df = stats_data_melted[stats_data_melted['Sample ID'] == st.session_state.fsf_spectrum_select]
                    x, y = filtered_fsf_df['Ramanshift'].to_numpy(), filtered_fsf_df['Intensity'].to_numpy()

                    if st.session_state.fsf_algorithm_version == "Discrete":
                        max_cofits = {
                            "Quick (m=6)": 6,
                            "Standard (m=9)": 9,
                            "Slow (m=12)": 12,
                            "Thorough (m=15)": 15
                        }[st.session_state.fsf_runtime_control]

                        start = time.perf_counter()
                        fit, residual, components, rmse = function.fit_full_spectrum_v2(x, y, 
                                                                        num_peaks=st.session_state.fsf_num_peaks,
                                                                        cofit_range_multiplier=st.session_state.fsf_cofit_range_multiplier,
                                                                        peak_shape=st.session_state.fsf_peak_shape,
                                                                        max_cofits=max_cofits
                                                                    )
                        end = time.perf_counter()
                        st.session_state.fsf_time = end - start # fit runtime

                        f_params={
                            "algorithm":"discrete",
                            "num_peaks":st.session_state.fsf_num_peaks,
                            "peak_shape":st.session_state.fsf_peak_shape,
                            "cofit_range_multiplier":st.session_state.fsf_cofit_range_multiplier,
                            "runtime":st.session_state.fsf_time,
                            "runtime_option":st.session_state.fsf_runtime_control,
                        }

                    elif st.session_state.fsf_algorithm_version == "Prominence-Based":
                        start = time.perf_counter()
                        fit, residual, components, rmse = function.fit_full_spectrum_v3(x, y, 
                                                                        min_prominence=st.session_state.fsf_prominence_threshold,
                                                                        peak_shape=st.session_state.fsf_peak_shape,
                                                                    )
                        end = time.perf_counter()
                        st.session_state.fsf_time = end - start # fit runtime

                        f_params={
                            "algorithm":"prominence_based",
                            "min_prominence":st.session_state.fsf_prominence_threshold,
                            "peak_shape":st.session_state.fsf_peak_shape,
                            "runtime":st.session_state.fsf_time,
                        }

                    log.log_function_call("Analytics_Peak_Fitting_Full_Spectrum",
                                          f_params=f_params)

                    st.session_state.fsf_residual_df = pd.DataFrame(np.array([x, residual]).T, columns=['Ramanshift', 'Residual'])
                    st.session_state.fsf_rmse = rmse

                    st.session_state.fsf_component_params_df = pd.DataFrame([c['parameters'] for c in components])

                    st.session_state.fsf_num_components = len(components)
                    st.session_state.fsf_results_df_with_components = pd.DataFrame(np.array([x, y]+[c['curve'] for c in components]+[fit]).T, columns=['Ramanshift', st.session_state.fsf_spectrum_select]+[f"Component {i}" for i in range(st.session_state.fsf_num_components)]+['Total Fit'])
                    st.session_state.fsf_results_df = pd.DataFrame(np.array([x, y, fit]).T, columns=['Ramanshift', st.session_state.fsf_spectrum_select, 'Total Fit'])

                results_df = st.session_state.fsf_results_df_with_components #if st.session_state.fsf_plot_components else st.session_state.fsf_results_df
                results_df_melted = results_df.melt(id_vars=['Ramanshift'], var_name='Sample ID', value_name='Intensity').round(4)

                st.success(f"Fit completed in {st.session_state.fsf_time:.3f} seconds.")

                # Plot results

                component_base_filter = {
                    "Gaussian":0.1,
                    "Lorentzian":5.0,
                    "Pseudovoigt":1.0,
                }[st.session_state.fsf_peak_shape]

                fsf_plot = (alt.Chart(results_df_melted)
                    .transform_filter((alt.datum['Intensity'] >= component_base_filter) | (alt.datum['Sample ID'] == st.session_state.fsf_spectrum_select) | (alt.datum['Sample ID'] == "Total Fit"))
                    .mark_line()
                    .encode(
                        x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                        y=alt.Y('Intensity', title=analytics_y_axis_title, type='quantitative'),
                        tooltip=alt.value(None),
                        color=alt.Color("Sample ID:N", title="Sample", sort=[st.session_state.fsf_spectrum_select, "Total Fit"]),
                        size=alt.value(3)
                    ).properties(width=1300, height=400, title="Fit Spectrum - Total Fit")
                    .interactive()
                )

                residual_plot = (alt.Chart(st.session_state.fsf_residual_df)
                    .mark_line()
                    .encode(
                        x=alt.X('Ramanshift', title=analytics_x_axis_title, type='quantitative'),
                        y=alt.Y('Residual', title=analytics_y_axis_title, type='quantitative'),
                        tooltip=alt.value(None),
                        color=alt.value('red'),
                        size=alt.value(3)
                    ).properties(width=1300, height=300, title=f"Fit Spectrum - Residual / (RMSE = {st.session_state.fsf_rmse:.3f})")
                )
                st.altair_chart(function.style_altair_chart(fsf_plot), use_container_width=False)
                log.log_plot_generated_count()
                st.altair_chart(function.style_altair_chart(residual_plot), use_container_width=False)
                log.log_plot_generated_count()

                st.write("Component Parameters")
                st.dataframe(st.session_state.fsf_component_params_df.round(4))

                st.write("Component Curves")
                st.dataframe(st.session_state.fsf_results_df_with_components.round(4)[['Ramanshift', st.session_state.fsf_spectrum_select, 'Total Fit']+[f"Component {i}" for i in range(st.session_state.fsf_num_components)]])
