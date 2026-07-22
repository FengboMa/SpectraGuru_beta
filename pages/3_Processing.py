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
import log_utils as log
from auth_utils import force_login

force_login()

function.wide_space_default()

if 'preprocessing_log' not in st.session_state:
    st.session_state.preprocessing_log = []

DEFAULT_X_AXIS_TITLE = "Raman shift/cm⁻¹"
DEFAULT_Y_AXIS_TITLE = "Intensity/a.u."

if "custom_axis_titles_act" not in st.session_state:
    st.session_state.custom_axis_titles_act = False
if "custom_x_axis_title" not in st.session_state:
    st.session_state.custom_x_axis_title = DEFAULT_X_AXIS_TITLE
if "custom_y_axis_title" not in st.session_state:
    st.session_state.custom_y_axis_title = DEFAULT_Y_AXIS_TITLE

def store_widget_value(key):
    st.session_state[key] = st.session_state[f"_{key}"]


def load_widget_value(key):
    st.session_state[f"_{key}"] = st.session_state[key]


def format_preprocessing_parameters(parameters):
    label_overrides = {
        "fft_threshold": "FFT threshold",
        "padding_method": "Padding method",
        "window_length": "Window length",
        "polynomial_order": "Polynomial order",
        "window_size": "Window size",
        "num_iterations": "Number of iterations",
        "max_iter": "Maximum iterations",
        "single_threshold": "Single threshold",
        "distance_threshold": "Distance threshold",
        "correlation_threshold": "Correlation threshold",
        "fitting_ranges": "Fitting ranges",
        "zap_length": "Zap length",
        "window_start": "Window start",
        "window_end": "Window end",
        "porder": "P-order",
        "lambda": "Lambda",
        "p": "P",
        "d": "D",
        "tau": "Tau",
        "min": "Min",
        "max": "Max",
    }
    parts = []
    for key, value in parameters.items():
        label = label_overrides.get(key, key.replace('_', ' ').capitalize())
        parts.append(f"{label}: {value}")
    return ", ".join(parts)


def render_preprocessing_log(log_entries):
    if not log_entries:
        st.write("**Preprocessing Steps Applied**")
        st.write("No preprocessing applied.")
        return

    st.write("**Preprocessing Steps Applied**")
    for idx, entry in enumerate(log_entries, start=1):
        params_text = format_preprocessing_parameters(entry["parameters"])
        st.write(f"{idx}. {entry['display_name']}, {params_text}")


def build_preprocessing_log_line(log_entries):
    if not log_entries:
        return "No preprocessing applied."

    formatted_steps = []
    for idx, entry in enumerate(log_entries, start=1):
        params_text = format_preprocessing_parameters(entry["parameters"])
        formatted_steps.append(f"{idx}. {entry['display_name']}, {params_text}")
    return " | ".join(formatted_steps)


def collect_current_preprocessing_entries():
    run_log_entries = []

    if st.session_state.interpolation_act:
        run_log_entries.append({
            "step": "interpolation",
            "display_name": "Interpolation",
            "parameters": {"parameter": "none"}
        })

    if st.session_state.crop_act:
        crop_min = min(st.session_state.crop_min, st.session_state.crop_max)
        crop_max = max(st.session_state.crop_min, st.session_state.crop_max)
        run_log_entries.append({
            "step": "crop",
            "display_name": "Crop",
            "parameters": {"min": crop_min, "max": crop_max}
        })

    if st.session_state.despike_act:
        if st.session_state.despike_function == "Auto despike method":
            run_log_entries.append({
                "step": "despike",
                "display_name": "Despike",
                "parameters": {
                    "function": "Auto despike method",
                    "threshold": st.session_state.despike_act_threshold,
                    "zap_length": st.session_state.despike_act_zap_length
                }
            })
        elif st.session_state.despike_function == "Manual despike method":
            if "despike_applied_range" not in st.session_state:
                raise AttributeError("despike_applied_range")
            run_log_entries.append({
                "step": "despike",
                "display_name": "Despike",
                "parameters": {
                    "function": "Manual despike method",
                    "threshold": st.session_state.despike_act_threshold,
                    "zap_length": st.session_state.despike_act_zap_length,
                    "window_start": st.session_state.despike_applied_range[0],
                    "window_end": st.session_state.despike_applied_range[1]
                }
            })

    if st.session_state.smoothening_act:
        if st.session_state.smoothening_function == "Savitzky-Golay filter":
            run_log_entries.append({
                "step": "smoothening",
                "display_name": "Smoothing",
                "parameters": {
                    "function": "Savitzky-Golay filter",
                    "window_length": st.session_state.smoothening_act_window_length,
                    "polynomial_order": st.session_state.smoothening_act_polyorder
                }
            })
        elif st.session_state.smoothening_function == "1D Fast Fourier Transform filter":
            run_log_entries.append({
                "step": "smoothening",
                "display_name": "Smoothing",
                "parameters": {
                    "function": "1D Fast Fourier Transform filter",
                    "fft_threshold": st.session_state.smoothening_act_FFT_threshold,
                    "padding_method": st.session_state.smoothening_act_FFT_padding
                }
            })
        elif st.session_state.smoothening_function == "Median filter":
            run_log_entries.append({
                "step": "smoothening",
                "display_name": "Smoothing",
                "parameters": {
                    "function": "Median filter",
                    "window_size": st.session_state.smoothening_act_median_filter_window_size,
                    "padding_method": st.session_state.smoothening_act_median_filter_padding_method
                }
            })
        elif st.session_state.smoothening_function == "Wavelet Denoising":
            method = st.session_state.wavelet_method
            params = {
                "function": "Wavelet Denoising",
                "method": method,
                "wavelet": st.session_state.wavelet_family,
                "level": st.session_state.wavelet_level
            }
            if method in ("Sardy Block Coordinate Relaxation(BCR)", "Sardy BCR"):
                params["n_iter"] = st.session_state.wavelet_n_iter
                params["loss"] = st.session_state.wavelet_loss
            else:
                params["mode"] = st.session_state.wavelet_mode
            run_log_entries.append({
                "step": "smoothening",
                "display_name": "Smoothing",
                "parameters": params
            })

    if st.session_state.baselineremoval_act:
        if st.session_state.baselineremoval_function == "airPLS":
            run_log_entries.append({
                "step": "baseline_removal",
                "display_name": "Baseline Removal",
                "parameters": {
                    "function": "airPLS",
                    "lambda": st.session_state.baselineremoval_airPLS_lambda,
                    "porder": st.session_state.baselineremoval_airPLS_porder,
                    "itermax": st.session_state.baselineremoval_airPLS_itermax,
                    "tau": st.session_state.baselineremoval_airPLS_tau
                }
            })
        if st.session_state.baselineremoval_function == "ModPoly":
            run_log_entries.append({
                "step": "baseline_removal",
                "display_name": "Baseline Removal",
                "parameters": {
                    "function": "ModPoly",
                    "degree": st.session_state.baselineremoval_ModPoly_degree
                }
            })
        if st.session_state.baselineremoval_function == "Gaussian-Lorentzian Fitting":
            if "fitting_ranges" not in st.session_state:
                raise AttributeError("fitting_ranges")
            run_log_entries.append({
                "step": "baseline_removal",
                "display_name": "Baseline Removal",
                "parameters": {
                    "function": "Gaussian-Lorentzian Fitting",
                    "fitting_ranges": st.session_state.fitting_ranges
                }
            })
        if st.session_state.baselineremoval_function == "SNIP":
            run_log_entries.append({
                "step": "baseline_removal",
                "display_name": "Baseline Removal",
                "parameters": {
                    "function": "SNIP",
                    "num_iterations": st.session_state.baselineremoval_SNIP_num_iterations
                }
            })
        if st.session_state.baselineremoval_function in ("Asymmetric Least Squares(ALS)", "ALS"):
            run_log_entries.append({
                "step": "baseline_removal",
                "display_name": "Baseline Removal",
                "parameters": {
                    "function": "Asymmetric Least Squares(ALS)",
                    "lambda": st.session_state.baselineremoval_ALS_lambda,
                    "p": st.session_state.baselineremoval_ALS_p,
                    "d": st.session_state.baselineremoval_ALS_d,
                    "max_iter": st.session_state.baselineremoval_ALS_max_iter
                }
            })


    if st.session_state.normalization_act:
        run_log_entries.append({
            "step": "normalization",
            "display_name": "Normalization",
            "parameters": {
                "function": st.session_state.normalization_function,
                "parameter": "none"
            }
        })

    if st.session_state.outlierremoval_act:
        run_log_entries.append({
            "step": "outlier_removal",
            "display_name": "Outlier Removal",
            "parameters": {
                "single_threshold": st.session_state.outlierremoval_act_single_threshold,
                "distance_threshold": st.session_state.outlierremoval_act_distance_threshold,
                "correlation_threshold": st.session_state.outlierremoval_act_correlation_threshold
            }
        })

    return run_log_entries


def apply_preprocessing_step(df, step_entry):
    step = step_entry["step"]
    params = step_entry["parameters"]
    result_df = df.copy()
    remove_outliers_log = None

    if step == "interpolation":
        interpolated_df = pd.DataFrame(result_df.iloc[:, 0].round(), columns=[result_df.columns[0]])
        for col in result_df.columns[1:]:
            interpolator = interp1d(
                result_df.iloc[:, 0],
                result_df[col],
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            interpolated_df[col] = interpolator(result_df.iloc[:, 0].round())
        return interpolated_df.drop_duplicates(), remove_outliers_log

    if step == "crop":
        return result_df[
            (result_df.iloc[:, 0] >= params["min"])
            & (result_df.iloc[:, 0] <= params["max"])
        ], remove_outliers_log

    if step == "despike":
        if params["function"] == "Auto despike method":
            result_df.iloc[:, 1:] = function.despikeSpec(
                spectra=result_df.iloc[:, 1:],
                ramanshift=result_df.iloc[:, 0],
                threshold=params["threshold"],
                zap_length=params["zap_length"]
            )
        elif params["function"] == "Manual despike method":
            result_df.iloc[:, 1:] = function.despikeSpec_v2(
                spectra=result_df.iloc[:, 1:],
                ramanshift=result_df.iloc[:, 0],
                threshold=params["threshold"],
                zap_length=params["zap_length"],
                window_start=params["window_start"],
                window_end=params["window_end"]
            )
        return result_df, remove_outliers_log

    if step == "smoothening":
        if params["function"] == "Savitzky-Golay filter":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                lambda col: function.savgol_filter_spectra(
                    col,
                    window_length=params["window_length"],
                    polyorder=params["polynomial_order"]
                )
            )
        elif params["function"] == "1D Fast Fourier Transform filter":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                lambda col: function.FFT_spectra(
                    col,
                    FFT_threshold=params["fft_threshold"],
                    padding_method=params["padding_method"]
                )
            )
        elif params["function"] == "Median filter":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                lambda col: function.median_filter_spectra(
                    col,
                    window_size=params["window_size"],
                    padding_method=params["padding_method"]
                )
            )
        elif params["function"] == "Wavelet Denoising":
            if params["method"] in ("Sardy Block Coordinate Relaxation(BCR)", "Sardy BCR"):
                result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                    lambda col: function.wavelet_denoise_sardy(
                        col,
                        wavelet=params["wavelet"],
                        level=params["level"],
                        n_iter=params["n_iter"],
                        loss=params["loss"]
                    )
                )
            elif params["method"] in ("Standard Universal Thresholding", "Standard"):
                result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                    lambda col: function.wavelet_denoise_standard(
                        col,
                        wavelet=params["wavelet"],
                        level=params["level"],
                        mode=params["mode"]
                    )
                )
        return result_df, remove_outliers_log

    if step == "baseline_removal":
        if params["function"] == "airPLS":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:] - result_df.iloc[:, 1:].apply(
                lambda col: function.airPLS(
                    col.values,
                    lambda_=params["lambda"],
                    porder=params["porder"],
                    itermax=params["itermax"],
                    tau=params["tau"]
                )
            )
        elif params["function"] == "ModPoly":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:] - result_df.iloc[:, 1:].apply(
                lambda col: function.ModPoly(col.values, degree=params["degree"])
            )
        elif params["function"] == "Gaussian-Lorentzian Fitting":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:] - result_df.iloc[:, 1:].apply(
                lambda col: function.GLF(
                    col.values,
                    wavenumber=result_df.iloc[:, 0].values,
                    fitting_ranges=params["fitting_ranges"]
                )
            )
        elif params["function"] == "SNIP":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:] - result_df.iloc[:, 1:].apply(
                lambda col: function.snip_1d(
                    col.values,
                    iterations=params["num_iterations"],
                    use_lls=True, # hardcoded
                    poly_window=15, # hardcoded
                    poly_deg=1, # hardcoded
                    return_baseline=True # hardcoded
                )
            )
        elif params["function"] in ("Asymmetric Least Squares(ALS)", "ALS"):
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                lambda col: function.als_baseline_removal(
                    col.values,
                    lam=params["lambda"],
                    p=params["p"],
                    d=params["d"],
                    max_iter=params["max_iter"]
                )
            )
        return result_df, remove_outliers_log

    if step == "normalization":
        if params["function"] == "Normalize by area":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(
                function.normalize_by_area,
                ramanshift=result_df.iloc[:, 0],
                axis=0
            )
        elif params["function"] == "Normalize by peak":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(function.normalize_by_peak, axis=0)
        elif params["function"] == "Min max normalize":
            result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(function.min_max_normalize, axis=0)
        return result_df, remove_outliers_log

    if step == "outlier_removal":
        df_cleaned, remove_outliers_log = function.remove_outliers(
            result_df,
            single_thresh=params["single_threshold"],
            distance_thresh=params["distance_threshold"],
            coeff_thresh=params["correlation_threshold"]
        )
        return pd.concat([result_df.iloc[:, 0], df_cleaned], axis=1), remove_outliers_log

    return result_df, remove_outliers_log


def rebuild_dataframe_from_log(log_entries=None):
    if log_entries is None:
        log_entries = st.session_state.preprocessing_log

    rebuilt_df = st.session_state.backup.copy()
    latest_remove_outliers_log = None

    for step_entry in log_entries:
        rebuilt_df, step_remove_outliers_log = apply_preprocessing_step(rebuilt_df, step_entry)
        if step_remove_outliers_log is not None:
            latest_remove_outliers_log = step_remove_outliers_log

    st.session_state.df = rebuilt_df
    if latest_remove_outliers_log is None:
        st.session_state.pop("remove_outliers_log", None)
    else:
        st.session_state.remove_outliers_log = latest_remove_outliers_log


if st.session_state.pop("undo_pending", False):
    if st.session_state.preprocessing_log:
        st.session_state.preprocessing_log.pop()
        function.clear_processing_toggles()
        rebuild_dataframe_from_log()
        st.session_state.refresh_plot_pending = True
    else:
        st.session_state.no_undo_available = True

if st.session_state.pop("no_undo_available", False):
    st.toast("No preprocessing step to undo.", icon="⚠️")

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

        interpolation_act = st.toggle("Interpolation", value=False, help="Round each Raman shift value to the closest integer.", key='interpolation_act')
        # st.sidebar.write(interpolation_ref_x)
        
        # crop
        # st.sidebar.markdown("**Crop**")
        c_crop_min = float(st.session_state.df.iloc[:, 0].min())
        c_crop_max = float(st.session_state.df.iloc[:, 0].max())
        if 'crop_act' not in st.session_state:
            st.session_state.crop_act = False
        crop_act = st.toggle("Crop", value=False, help="Select the Raman-shift range to keep.", key='crop_act')
        
        if crop_act:
            st.write("Spectra range: " ,c_crop_min, " - ", c_crop_max)
            st.number_input("Crop min", 
                                    min_value=c_crop_min, 
                                    max_value=c_crop_max,
                                    value=c_crop_min,
                                    step=1.00,
                                    key="crop_min")
            
            st.number_input("Crop max", 
                                    min_value=c_crop_min, 
                                    max_value=c_crop_max,
                                    value=c_crop_max,
                                    step=1.00,
                                    key="crop_max")
        
        # Despike
        # st.sidebar.markdown("**Despike**")
        
        if 'despike_act' not in st.session_state:
            st.session_state.despike_act = False
        
        despike_act = st.toggle("Despike",
                                        value=False,
                                        help="**Auto despike method** - Automatically detects and corrects spikes across the entire spectrum. Regions where the signal exceeds a defined threshold within a specified scan width are replaced with linear interpolation. This method may slightly alter the overall spectrum. \n \n **Manual despike method** - Allows users to define a specific window where despiking is applied. Only spikes within this region are corrected, minimizing unintended effects on the rest of the spectrum.",
                                        key='despike_act')
        
        if despike_act:
            
            st.session_state.despike_function = st.selectbox(label="Select despike function",  options=["Auto despike method","Manual despike method"])
            
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
                with st.form("despike_fitting_range_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        start = st.number_input(f"Start of range", key=f"despike_start", step=1.0, format="%.2f",value=wavenumber_min)
                    with col2:
                        end = st.number_input(f"End of range", key=f"despike_end", step=1.0, format="%.2f", value=wavenumber_max)

                    submitted = st.form_submit_button("Apply fitting ranges")
                    if submitted:
                        # Clip start if needed
                        if start < wavenumber_min:
                            st.warning(f"Start value clipped from {start:.2f} to {wavenumber_min:.2f}")
                            start = wavenumber_min

                        # Clip end if needed
                        if end > wavenumber_max:
                            st.warning(f"End value clipped from {end:.2f} to {wavenumber_max:.2f}")
                            end = wavenumber_max

                        # Validate order
                        if start >= end:
                            st.error("Start value must be less than end value.")
                        else:
                            st.session_state.despike_applied_range = (start, end)
                            st.success(f"Fitting range ({start:.2f}, {end:.2f}) applied.")
        
        # Smoothing
        # st.sidebar.markdown("**Smoothing**")
        
        if 'smoothening_act' not in st.session_state:
            st.session_state.smoothening_act = False
        
        smoothening_act = st.toggle("Smoothing",
                                        value=False,
                                        help="Smoothing in spectra processing reduces noise and enhances the signal by averaging adjacent data points to produce a clearer representation of the spectral data.",
                                        key='smoothening_act')

        if smoothening_act:
            # Add more functions to this selectbox if needed
            st.session_state.smoothening_function = st.selectbox(
                label="Select smoothing function",
                options=["Savitzky-Golay filter","1D Fast Fourier Transform filter", "Median filter", "Wavelet Denoising"],
                format_func={
                    "Savitzky-Golay filter": "Savitzky-Golay filter",
                    "1D Fast Fourier Transform filter": "1D Fast Fourier Transform filter",
                    "Median filter": "Median filter",
                    "Wavelet Denoising": "Wavelet denoising"
                }.get
            )
            
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
                The threshold sets the cutoff frequency for the low-pass filter applied to the spectra in the frequency domain. This threshold determines which frequency components are preserved and which are filtered out.
                
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
                smoothening_act_FFT_padding = st.selectbox(label="Select FFT padding method",  options=["mirror",
                                                                                                                    "edge",
                                                                                                                    "zero"],
                                                                key = "smoothening_act_FFT_padding",
                                                                help = help_txt2)
            elif st.session_state.smoothening_function == "Median filter":
                window_size_help = '''
                The window size parameter specifies the length of the window for the median filter to use. A small window size can remove sharp spikes or outliers while minimizing artifacts. Larger window sizes may result in feature loss and distortion of the original spectra.

                Window size must be odd. The max window size is 51, but smaller window sizes may still produce significant distortion and artifacts. Be sure to select an appropriate window size considering the width of features in your spectra.
                '''
                st.number_input(label="Window size", key = "smoothening_act_median_filter_window_size",
                                                min_value = 3, max_value = 51, value = 3,
                                                step=2, placeholder="Insert a number", help=window_size_help)
                padding_help = '''
                The padding method parameter specifies the method used to pad the signal before applying the median filter. Padding helps to reduce edge effects and minimize artifacts introduced by the filtering process.
                
                **Mirror Padding ('mirror'):** Reflects the signal at its edges, creating a smooth transition.
                
                **Edge Padding ('edge'):** Repeats the edge values of the signal.
                
                **Zero Padding ('zero'):** Adds zeros to the edges of the signal. May introduce artifacts at the edges.
                '''
                st.selectbox(label="Padding method",
                             options=["mirror", "edge", "zero"],
                             key="smoothening_act_median_filter_padding_method",
                             help=padding_help)
            elif st.session_state.smoothening_function == "Wavelet Denoising":
                wavelet_methods = ["Sardy Block Coordinate Relaxation(BCR)", "Standard Universal Thresholding"]
                wavelet_help = '''
                Selects the coefficient-shrinkage method used after wavelet decomposition.

                Sardy Block Coordinate Relaxation(BCR) applies an iterative robust shrinkage procedure to detail coefficients. Standard Universal Thresholding applies a single universal threshold to detail coefficients.

                Default: Sardy Block Coordinate Relaxation(BCR).
                '''
                st.selectbox(
                    label="Wavelet denoising method",
                    options=wavelet_methods,
                    format_func={
                        "Sardy Block Coordinate Relaxation(BCR)": "Sardy block coordinate relaxation (BCR)",
                        "Standard Universal Thresholding": "Standard universal thresholding"
                    }.get,
                    help=wavelet_help,
                    key="wavelet_method"
                )
                st.selectbox(
                    label="Wavelet family",
                    options=["sym4", "sym8", "db4", "db8", "coif1", "coif3", "haar"],
                    index=0,
                    help="Selects the wavelet basis used to decompose and reconstruct each spectrum. The options define different compact-support wavelet shapes. Default: sym4.",
                    key="wavelet_family"
                )
                st.selectbox(
                    label="Decomposition level",
                    options=[1, 2, 3, 4, 5, 6],
                    index=3,
                    help="Sets how many recursive wavelet decomposition levels are applied before coefficient shrinkage. Larger values separate broader signal structures into lower-frequency components. Default: 4.",
                    key="wavelet_level"
                )
                if st.session_state.get("wavelet_method") in ("Sardy Block Coordinate Relaxation(BCR)", "Sardy BCR"):
                    st.number_input(
                        label="BCR iterations",
                        min_value=1,
                        max_value=15,
                        value=10,
                        step=1,
                        help="Sets the maximum number of block coordinate relaxation passes used to update the shrinkage solution. Larger values allow additional refinement and increase runtime. Default: 10.",
                        key="wavelet_n_iter"
                    )
                    st.selectbox(
                        label="Robust loss function",
                        options=["huber", "l1"],
                        index=0,
                        help="Selects the residual loss used by the robust iterative shrinkage method. Huber uses quadratic behavior near zero residuals and linear behavior for large residuals; l1 uses absolute residual magnitude. Default: huber.",
                        key="wavelet_loss"
                    )
                elif st.session_state.get("wavelet_method") in ("Standard Universal Thresholding", "Standard"):
                    st.selectbox(
                        label="Thresholding mode",
                        options=["soft", "hard"],
                        index=0,
                        help="Selects how coefficients below the universal threshold are treated. Soft thresholding shrinks retained coefficients toward zero; hard thresholding keeps retained coefficients unchanged. Default: soft.",
                        key="wavelet_mode"
                    )
        # Baseline removal
        # st.markdown("**Baseline Removal**")
        
        if 'baselineremoval_act' not in st.session_state:
            st.session_state.baselineremoval_act = False
        
        baselineremoval_act = st.toggle("Baseline removal",
                                                value=False,
                                                help="Remove baselines or backgrounds by fitting a baseline function or subtracting a baseline estimate from the data.",
                                                key='baselineremoval_act')
        
        if baselineremoval_act:
            # Add more functions to this selectbox if needed
            baselineremoval_functions = ["airPLS", "ModPoly","Gaussian-Lorentzian Fitting", "SNIP", "ALS"]
            st.session_state.baselineremoval_function = st.selectbox(
                label="Select baseline removal function",
                options=baselineremoval_functions,
                format_func={
                    "airPLS": "AirPLS",
                    "ModPoly": "ModPoly",
                    "Gaussian-Lorentzian Fitting": "Gaussian-Lorentzian fitting",
                    "SNIP": "SNIP",
                    "ALS": "ALS"
                }.get
            )
            
            if st.session_state.baselineremoval_function == "airPLS":
                st.session_state.baselineremoval_airPLS_lambda = st.number_input(label="AirPLS lambda", help="The larger lambda is,  the smoother the resulting background, z.",
                                                                        min_value = 1, max_value = 1000000, value = 100,
                                                                        step = 1, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_porder = st.number_input(label="AirPLS p-order", help="Adaptive iteratively reweighted penalized least squares for baseline fitting.",
                                                                        min_value=1, max_value = 10, value = 1, 
                                                                        step = 1, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_itermax = st.number_input(label="AirPLS maximum iterations",
                                                                        min_value=5, max_value = 1000, value = 15, 
                                                                        step = 5, placeholder="Insert a number")
                
                st.session_state.baselineremoval_airPLS_tau = st.number_input(label="AirPLS tolerance",
                                                                        min_value=0.0000000001, max_value = 0.100000000, value = 0.001000000, 
                                                                        step = 0.0000000001, placeholder="Insert a number",format="%.10f") 
                
            elif st.session_state.baselineremoval_function == "ModPoly":
                st.session_state.baselineremoval_ModPoly_degree = st.number_input(label="ModPoly polynomial degree",
                                                                        min_value=1, max_value = 20, value = 5, 
                                                                        step = 1, placeholder="Insert a number") 
            elif st.session_state.baselineremoval_function == "Gaussian-Lorentzian Fitting":
                st.session_state.baselineremoval_GLF_num_range = st.number_input(label="Number of fitting ranges",
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
            elif st.session_state.baselineremoval_function == "SNIP":
                st.session_state.baselineremoval_SNIP_num_iterations = st.number_input(label="SNIP Iterations",
                                                                            help= "Determines the maximum peak width to be removed. Higher values create a smoother, lower baseline by allowing the algorithm to 'clip' wider peaks.",
                                                                            min_value=10, max_value = 200, value = 50,
                                                                            step = 1, placeholder="Insert a number")
            elif st.session_state.baselineremoval_function in ("Asymmetric Least Squares(ALS)", "ALS"):
                st.session_state.baselineremoval_ALS_lambda = st.number_input(
                    label="ALS lambda",
                    help="Controls the smoothness penalty for the estimated baseline. Larger values penalize curvature more strongly and produce a smoother baseline. Default: 100.",
                    min_value=1.0,
                    max_value=1e10,
                    value=100.0,
                    step=1e4,
                    format="%.0f",
                    placeholder="Insert a number"
                )
                st.session_state.baselineremoval_ALS_p = st.number_input(
                    label="ALS asymmetry (p)",
                    help="Controls the asymmetric weighting between positive and negative residuals during baseline fitting. Smaller values weight positive peak residuals less, placing the fitted baseline below peaks. Default: 0.001.",
                    min_value=0.000001,
                    max_value=0.999999,
                    value=0.001,
                    step=0.0001,
                    format="%.6f",
                    placeholder="Insert a number"
                )
                st.session_state.baselineremoval_ALS_d = st.number_input(
                    label="ALS difference order (d)",
                    help="Sets the finite-difference order used in the smoothness penalty. Order 1 penalizes slope changes; order 2 penalizes curvature; order 3 applies a higher-order curvature penalty. Default: 1.",
                    min_value=1,
                    max_value=3,
                    value=1,
                    step=1,
                    placeholder="Insert a number"
                )
                st.session_state.baselineremoval_ALS_max_iter = st.number_input(
                    label="ALS maximum iterations",
                    help="Sets the maximum number of reweighted least-squares updates used to estimate the baseline. Larger values allow more weight updates and increase runtime. Default: 50.",
                    min_value=1,
                    max_value=200,
                    value=50,
                    step=1,
                    placeholder="Insert a number"
                )
        # Normalization
        # st.markdown("**Normalization**")
        
        if 'normalization_act' not in st.session_state:
            normalization_act = False
        
        normalization_act = st.toggle("Normalization", 
                                                value=False, 
                                                help="Normalize by Area divides each spectrum's values by the area under the spectrum, effectively setting the area under each spectrum to a common value.", 
                                                key='normalization_act')
        
        if normalization_act:
            # Add more functions to this selectbox if needed
            st.session_state.normalization_function = st.selectbox(label="Select normalization function",  options=["Normalize by area",
                                                                                                                "Normalize by peak",
                                                                                                                "Min max normalize"],
                                                                    format_func={
                                                                        "Normalize by area": "Normalize by area",
                                                                        "Normalize by peak": "Normalize by peak",
                                                                        "Min max normalize": "Min-max normalization"
                                                                    }.get)

        # Outlier removal
        if 'outlierremoval_act' not in st.session_state:
            outlierremoval_act = False

        outlierremoval_act = st.toggle("Outlier removal",
                                        value=False, 
                                        help="The function removes outlier spectra from a dataframe based on single threshold, distance, and correlation criteria.", 
                                        key='outlierremoval_act')
        
        if outlierremoval_act:
            # Add more functions to this selectbox if needed
            st.session_state.outlierremoval_act_single_threshold = st.number_input(label="Outlier removal single threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 4.00,
                                                            step = 0.01, placeholder="Insert a number")
            
            st.session_state.outlierremoval_act_distance_threshold = st.number_input(label="Outlier removal distance threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 6.00,
                                                            step = 0.01, placeholder="Insert a number")
            
            st.session_state.outlierremoval_act_correlation_threshold = st.number_input(label="Outlier removal correlation threshold",
                                                            min_value = 0.01, max_value = 20.00, value = 4.00,
                                                            step = 0.01, placeholder="Insert a number")
    with st.sidebar:
        pre_processing()
            
    # Processing button and reaction
    if st.sidebar.button("Process", type='primary', key = 'process'):
        log.log_spectra_processed_count()
        try:
            run_log_entries = collect_current_preprocessing_entries()
        except AttributeError as e:
            if "fitting_ranges" in str(e):
                st.error("⚠️ Please go to the sidebar and set and apply your fitting ranges before applying Gaussian-Lorentzian fitting.")
                run_log_entries = []
            elif "despike_applied_range" in str(e):
                st.error("⚠️ Please apply your manual despike range before processing.")
                run_log_entries = []
            else:
                raise e
        for step_entry in run_log_entries:
            if step_entry["step"] == "despike":
                if step_entry["parameters"]["function"] == "Auto despike method":
                    log.log_function_call("Processing_Despike_Auto", f_params={
                        'threshold': step_entry["parameters"]["threshold"],
                        'zap_length': step_entry["parameters"]["zap_length"]
                    })
                else:
                    log.log_function_call("Processing_Despike_Manual", f_params={
                        'threshold': step_entry["parameters"]["threshold"],
                        'zap_length': step_entry["parameters"]["zap_length"],
                        'window_start': step_entry["parameters"]["window_start"],
                        'window_end': step_entry["parameters"]["window_end"]
                    })
            elif step_entry["step"] == "smoothening":
                if step_entry["parameters"]["function"] == "Savitzky-Golay filter":
                    log.log_function_call("Processing_Smoothing_Savgol_Filter", f_params={
                        'window_length': step_entry["parameters"]["window_length"],
                        'polyorder': step_entry["parameters"]["polynomial_order"]
                    })
                elif step_entry["parameters"]["function"] == "1D Fast Fourier Transform filter":
                    log.log_function_call("Processing_Smoothing_FFT_Filter", f_params={
                        'FFT_threshold': step_entry["parameters"]["fft_threshold"],
                        'padding_method': step_entry["parameters"]["padding_method"]
                    })
                elif step_entry["parameters"]["function"] == "Median filter":
                    log.log_function_call("Processing_Smoothing_Median_Filter", f_params={
                        'window_size': step_entry["parameters"]["window_size"],
                        'padding_method': step_entry["parameters"]["padding_method"]
                    })
                elif step_entry["parameters"]["function"] == "Wavelet Denoising":
                    wavelet_params = {
                        'method': step_entry["parameters"]["method"],
                        'wavelet': step_entry["parameters"]["wavelet"],
                        'level': step_entry["parameters"]["level"]
                    }
                    if step_entry["parameters"]["method"] in ("Sardy Block Coordinate Relaxation(BCR)", "Sardy BCR"):
                        wavelet_params.update({
                            'n_iter': step_entry["parameters"]["n_iter"],
                            'loss': step_entry["parameters"]["loss"]
                        })
                    else:
                        wavelet_params.update({
                            'mode': step_entry["parameters"]["mode"]
                        })
                    log.log_function_call("Processing_Smoothing_Wavelet_Denoising", f_params=wavelet_params)
            elif step_entry["step"] == "baseline_removal":
                if step_entry["parameters"]["function"] == "airPLS":
                    log.log_function_call("Processing_Baseline_AirPLS", f_params={
                        'lambda_': step_entry["parameters"]["lambda"],
                        'porder': step_entry["parameters"]["porder"],
                        'itermax': step_entry["parameters"]["itermax"],
                        'tau': step_entry["parameters"]["tau"]
                    })
                elif step_entry["parameters"]["function"] == "ModPoly":
                    log.log_function_call("Processing_Baseline_Mod_Poly", f_params={
                        'degree': step_entry["parameters"]["degree"]
                    })
                elif step_entry["parameters"]["function"] == "SNIP":
                    log.log_function_call("Processing_Baseline_SNIP", f_params={
                        'num_iterations': step_entry["parameters"]["num_iterations"]
                    })
                elif step_entry["parameters"]["function"] in ("Asymmetric Least Squares(ALS)", "ALS"):
                    log.log_function_call("Processing_Baseline_ALS", f_params={
                        'lambda': step_entry["parameters"]["lambda"],
                        'p': step_entry["parameters"]["p"],
                        'd': step_entry["parameters"]["d"],
                        'max_iter': step_entry["parameters"]["max_iter"]
                    })
                else:
                    log.log_function_call("Processing_Baseline_Gaussian_Lorentzian_Fitting", f_params={
                        'fitting_ranges': step_entry["parameters"]["fitting_ranges"]
                    })
            elif step_entry["step"] == "normalization":
                if step_entry["parameters"]["function"] == "Normalize by area":
                    log.log_function_call("Processing_Normalization_Area", f_params={})
                elif step_entry["parameters"]["function"] == "Normalize by peak":
                    log.log_function_call("Processing_Normalization_Peak", f_params={})
                else:
                    log.log_function_call("Processing_Normalization_Minmax", f_params={})
            elif step_entry["step"] == "outlier_removal":
                log.log_function_call("Processing_Remove_Outliers", f_params={
                    'single_thresh': step_entry["parameters"]["single_threshold"],
                    'distance_thresh': step_entry["parameters"]["distance_threshold"],
                    'coeff_thresh': step_entry["parameters"]["correlation_threshold"]
                })

        if run_log_entries:
            candidate_log = st.session_state.preprocessing_log + run_log_entries
            try:
                rebuild_dataframe_from_log(candidate_log)
                st.session_state.preprocessing_log = candidate_log
            except Exception as e:
                failed_step = run_log_entries[-1]["display_name"] if run_log_entries else "Preprocessing"
                st.toast(f"{failed_step} could not be applied. Details: {e}", icon="⚠️")

    if st.sidebar.button("Undo", type='secondary', key='undo'):
        if st.session_state.preprocessing_log:
            st.session_state.undo_pending = True
            st.rerun()
        else:
            st.toast("No preprocessing step to undo.", icon="⚠️")
        
    
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
        valid_selected_spectra = [col for col in selected_spectra if col in df.columns]
        columns_to_select = [df.columns[0]] + valid_selected_spectra
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
        refresh_plot_pending = st.session_state.pop("refresh_plot_pending", False)
        plot_button_col, fast_mode_col, custom_axis_col = st.columns([0.16, 0.44, 0.40])
        if plot_button_col.button("Plot", type="primary", key="plot") or st.session_state.get("process") or refresh_plot_pending or st.session_state.get("reset"):
            st.session_state.temp = get_selected_columns(st.session_state.df, st.session_state.spectra_selected)

    
    # st.write(st.session_state.temp)
    # if st.button('end time'):
    #     st.session_state.elapsed_time = time.time() - st.session_state.start_time
    #     st.write("Test time (ms):")
    #     st.write(int(st.session_state.elapsed_time * 1000))
    # plot_row = row([0.1, 0.9])
    mode_option = fast_mode_col.toggle(label='Activate fast mode plotting', value=st.session_state['update_mode_option'], key='mode_option', help='Enable fast mode plotting for faster plotting times by sacrificing interactive functions. If you upload more than 20 spectra, fast mode will be activated automatically.')
    
    load_widget_value("custom_axis_titles_act")
    custom_axis_col.toggle(
        label="Custom axis titles",
        key="_custom_axis_titles_act",
        on_change=store_widget_value,
        args=("custom_axis_titles_act",),
        help="Enable custom X and Y axis labels for the plot and PNG export on this page."
    )

    if st.session_state.custom_axis_titles_act:
        load_widget_value("custom_x_axis_title")
        load_widget_value("custom_y_axis_title")
        axis_title_col1, axis_title_col2 = st.columns(2)
        with axis_title_col1:
            st.text_input(
                "Custom X title",
                key="_custom_x_axis_title",
                on_change=store_widget_value,
                args=("custom_x_axis_title",)
            )
        with axis_title_col2:
            st.text_input(
                "Custom Y title",
                key="_custom_y_axis_title",
                on_change=store_widget_value,
                args=("custom_y_axis_title",)
            )

    if st.session_state.custom_axis_titles_act:
        x_axis_title = st.session_state.custom_x_axis_title
        y_axis_title = st.session_state.custom_y_axis_title
    else:
        x_axis_title = DEFAULT_X_AXIS_TITLE
        y_axis_title = DEFAULT_Y_AXIS_TITLE
        
    
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
            def generate_altair_plot(data, x_axis_name, x_title, y_title):
                base = alt.Chart(data).mark_line().encode(
                x=alt.X(x_axis_name, title=x_title, type='quantitative'),
                y=alt.Y('Intensity', title=y_title, type='quantitative'),
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
            
            cached_plot = generate_altair_plot(data_melted, x_axis, x_axis_title, y_axis_title)
            # st.write("yyyyy")
            # Display Plot
            st.altair_chart(cached_plot, use_container_width=False)
            # st.write("zzzzz")
            log.log_plot_generated_count()
            
        elif mode_option == True:
                
            @st.cache_data
            # Generate the Altair plot
            def generate_altair_plot_fastmode(data, x_axis_name, x_title, y_title):
                base = base = alt.Chart(data).mark_line().encode(
                x=alt.X(x_axis_name, title=x_title, type='quantitative'),
                y=alt.Y('Intensity', title=y_title, type='quantitative'),
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
            
            cached_plot = generate_altair_plot_fastmode(data_melted, x_axis, x_axis_title, y_axis_title)
            # st.write("1111")
            # Display Plot
            st.altair_chart(cached_plot, use_container_width=False)
            # st.write("2222")
            log.log_plot_generated_count()
            # # Define the nearest selection
            # # click = alt.selection_single(nearest=True, on='click')
            # st.altair_chart(base, use_container_width=False)
            # function.log_plot_generated_count(st.session_state.log_file_path)
        
        
        # Download handlers
        @st.cache_data
        def download_df(df, export_timestamp, preprocessing_summary):
            csv_data = df.to_csv(index=False)
            metadata = [
                "# Data processed with SpectraGuru™",
                f"# Export timestamp: {export_timestamp}",
                f"# Preprocessing Steps Applied: {preprocessing_summary}",
                ""
            ]
            return ("\n".join(metadata) + csv_data).encode("utf-8")

        export_dt = datetime.now()
        export_timestamp = export_dt.isoformat(timespec="seconds")
        preprocessing_summary = build_preprocessing_log_line(st.session_state.preprocessing_log)
        csv = download_df(st.session_state.df, export_timestamp, preprocessing_summary)

        current_time = export_dt.strftime("%Y%m%d_%H%M%S")
        download_file_name = f"data_{current_time}.csv"
        download_plot_name = f"spectra_plot_{current_time}.png"

        png_bytes = function.make_matplotlib_png(
            data_melted,
            x_axis,
            x_label=x_axis_title,
            y_label=y_axis_title
        )

        # Create a single row with two columns
        dcol1,dcol2, col_spacer  = st.columns([1, 1, 3])

        with dcol1:
            st.download_button(
                label="Data export (CSV)",
                data=csv,
                file_name=download_file_name,
                mime="text/csv"
            )

        with dcol2:
            st.download_button(
                label="Plot export (PNG, 600 dpi)",
                data=png_bytes,
                file_name=download_plot_name,
                mime="image/png"
            )

        render_preprocessing_log(st.session_state.preprocessing_log)

        # Outlier removal log
        if st.session_state.outlierremoval_act:
            st.write("**The following spectra have been detected and removed by the outlier removal function**")
            st.table(st.session_state.remove_outliers_log)
                
    except:
        pass
