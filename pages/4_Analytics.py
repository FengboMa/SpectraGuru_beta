import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import altair as alt
# from streamlit_extras.chart_container import chart_container
from streamlit_extras.row import row
from datetime import datetime
import pandas as pd
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

if 'df' in st.session_state:
    st.sidebar.write("#### Analytics")

    # Side bar to select which plot and parameters
    st.sidebar.selectbox('Select Analytics Plot', 
                        options= ("Average Plot with Original Spectra", 
                                "Confidence Interval Plot",
                                "Spectra Derivation",
                                "Correlation Heatmap",
                                "Peak Identification and Stats",
                                "Hierarchically-clustered Heatmap",
                                "Principal Components Analysis (PCA)-Beta",
                                "T-SNE Dimensionality Reduction-Beta"),
                        key="stats_plot_select")

    if st.session_state.stats_plot_select == "Average Plot with Original Spectra":
        st.sidebar.toggle(label='Show spectra you selected', value=True, key = 'stats_avg_act',help='Show or hide original selected spectra.')
        st.sidebar.toggle(label='Show Standard Deviation', value=True, key = 'stats_avg_std_act',help='Show or hide Standard Deviation.')
    elif st.session_state.stats_plot_select == "Spectra Derivation":
        st.sidebar.selectbox(label="Normalization Method",
            options=("None", "Min-Max Normalization"),
            index=0,
            key="deriv_norm_method",
            help="Apply per-spectrum Min–Max scaling before taking derivatives.")
    elif st.session_state.stats_plot_select == "Correlation Heatmap":
        
        if st.sidebar.toggle(label='Customize Heatmap scale', value=False, key = 'heatmap_scale',help='Customize heatmap scale manually.'):
            st.sidebar.number_input(label='Heatmap scale min',min_value= -1.0, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_min',step = 0.01, value = 0.5, format="%.2f")
            st.sidebar.number_input(label='Heatmap scale max',min_value= -1.00, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_max',step = 0.01, value = 1.00, format="%.2f")
            
            if st.session_state.heatmap_min >= st.session_state.heatmap_max:
                st.sidebar.error('Invalid Number input.')
    elif st.session_state.stats_plot_select == "Peak Identification and Stats":
        
        st.sidebar.write("Find peaks only on average:")
        st.sidebar.write(True)
        
        st.sidebar.toggle(label="Auto Peak Identification", value=True, key="peak_iden_auto")
        
        if st.session_state.peak_iden_auto == True:
            st.session_state.peak_iden_height_p = 0
            st.session_state.peak_iden_threshold_p = 0
            st.session_state.peak_iden_distance_p = 1
            st.session_state.peak_iden_prominence_p = 0
            st.session_state.peak_iden_width_p = 0
            
            st.sidebar.number_input(label='Number of Peaks to identify',min_value= 1, max_value= 10000, placeholder='Insert a number',
                                        key = 'peak_iden_auto_num',step = 1, value = 10,
                                        help = "Number of peak you wish to identify automatically.")
            
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
    elif st.session_state.stats_plot_select == "Principal Components Analysis (PCA)-Beta":
        num_rows = st.session_state.df.shape[1] - 1
        pc_list = [f"PC{i+1}" for i in range(num_rows)] 
        st.sidebar.selectbox(label="Select Horizontal PC", options=pc_list, index=0,key="PCA_horizontal")
        st.sidebar.selectbox(label="Select Vertical PC", options=pc_list, index=1,key="PCA_vertical")
        st.sidebar.toggle(label="Coloring by setting labels", value=True, key="PCA_label")
    elif st.session_state.stats_plot_select == "T-SNE Dimensionality Reduction-Beta":
        max_perplexity = st.session_state.df.shape[1] - 1
        st.sidebar.select_slider(label="t-SNE Perplexity", options=list(range(1,max_perplexity)),value=2, key="tSNE_perplexity")
        st.sidebar.select_slider(label="t-SNE Maximum number of iterations", options=list(range(200,1001)), value=500, key="tSNE_n_iter")

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
        st.session_state.df_stats = st.session_state.temp
        st.session_state.df_stats['Average'] = st.session_state.df_stats.iloc[:, 1:].mean(axis=1)
        
        # st.write(st.session_state.df_stats)
        
        stats_data_melted = st.session_state.df_stats.melt(id_vars=['Ramanshift'], var_name='Sample ID', value_name='Intensity')
        
        # st.write(stats_data_melted)
        
        if st.session_state.stats_plot_select == "Average Plot with Original Spectra":
            
            if st.session_state.stats_avg_act:
                avg_stats_base = alt.Chart(stats_data_melted).mark_line().encode(
                        x=alt.X('Ramanshift', title='Raman shift/cm^-1', type='quantitative'),
                        y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
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
                function.log_plot_generated_count(st.session_state.log_file_path)
                
            else:
                filtered_avg_df = stats_data_melted[stats_data_melted['Sample ID'] == 'Average']
                avg_stats_base2 = alt.Chart(filtered_avg_df).mark_line().encode(
                        x=alt.X('Ramanshift', title='Raman shift/cm^-1', type='quantitative'),
                        y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
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
                function.log_plot_generated_count(st.session_state.log_file_path)
            
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
                    x='Ramanshift',
                    y='Standard Deviation'
                ).properties(
                            width=1300,
                            height=300,
                )
                
                combined_plot = alt.vconcat(show_plot, std_plot).resolve_scale(
                                                x='shared'  # Share the x-axis between the plots
                                            )
                combined_plot = function.style_altair_chart(combined_plot)
                st.altair_chart(combined_plot, use_container_width=False)
                function.log_plot_generated_count(st.session_state.log_file_path)
            else:
                show_plot = function.style_altair_chart(show_plot)    
                st.altair_chart(show_plot, use_container_width=False)
            
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
            
            # st.write(std_df)
            # Calculate mean and std deviation
            mean_values = std_df.mean(axis=1)
            std_values = std_df.std(axis=1)

            ci_upper = mean_values + std_values
            ci_lower = mean_values - std_values
            
            data = pd.DataFrame({
                'Ramanshift': ramanshift,
                'Mean': mean_values,
                'CI_Upper': ci_upper,
                'CI_Lower': ci_lower
            })
            
            base = alt.Chart(data).encode(
                x='Ramanshift'
            ).properties(
                            width=1300,
                            height=600,
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
            function.log_plot_generated_count(st.session_state.log_file_path)

        elif st.session_state.stats_plot_select == "Spectra Derivation":
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

                # Richer tip for users
                st.caption(
                    "**Tips for tuning Savitzky–Golay parameters:**\n"
                    "- Window length should be **odd**, typically between 3–25 for Raman/SERS spectra. "
                    "Use larger values for noisy data, smaller for narrow/sharp peaks.\n"
                    "- Polynomial order is usually **2 or 3**. "
                    "Higher order can follow sharper features but may also fit noise.\n"
                    "- Always ensure **poly < window length**. "
                    "- Try starting with `win=11`, `poly=3` and adjust if peaks look oversmoothed or too noisy."
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
                        x=alt.X("Ramanshift:Q", title="Raman shift / cm⁻¹"),
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
                        x=alt.X("Ramanshift:Q", title="Raman shift / cm⁻¹"),
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
                try:
                    function.log_function_use_count(st.session_state.function_log_file_path, "Spectra_Derived", st.session_state.df[1:].shape[1])
                    function.log_plot_generated_count(st.session_state.log_file_path)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Error during processing: {e}")
        
        elif st.session_state.stats_plot_select == "Correlation Heatmap":
            # Select only the columns we need for standard deviation calculation
            # Filter out the columns
            columns_to_include = [col for col in st.session_state.temp.columns if col not in ["Ramanshift", "Average","Standard Deviation"]]
            df_filtered = st.session_state.temp[columns_to_include]
            df_filtered['Average'] = df_filtered.mean(axis=1)
            
            # st.write(df_filtered)

            # Calculate the correlation matrix
            corr_matrix = df_filtered.corr()

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
            function.log_plot_generated_count(st.session_state.log_file_path)
            function.log_function_use_count(st.session_state.function_log_file_path, "Correlation_Heatmaps_Generated")
            
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
            filtered_avg_df = stats_data_melted[stats_data_melted['Sample ID'] == 'Average']
            
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
            
            peaks, properties = function.peak_identification(spectra=filtered_avg_df['Intensity'].to_numpy(),
                                                            height= st.session_state.peak_iden_height_p,
                                                            threshold = st.session_state.peak_iden_threshold_p,
                                                            distance = st.session_state.peak_iden_distance_p,
                                                            prominence = st.session_state.peak_iden_prominence_p,
                                                            width = st.session_state.peak_iden_width_p)


            # Extract Raman shift and intensity values
            raman_shift = filtered_avg_df['Ramanshift']
            intensity = filtered_avg_df['Intensity']

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
            peak_df = filtered_avg_df.iloc[peaks].copy()
            
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
            avg_stats_base2 = alt.Chart(filtered_avg_df).mark_line().encode(
                x=alt.X('Ramanshift', title='Raman shift/cm^-1', type='quantitative'),
                y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
                tooltip=alt.value(None),
                color=alt.value('blue'),
                size=alt.value(3)
            ).properties(
                width=1300,
                height=600,
                title='Spectra Average Data Plot'
            )

            # Step 4: Create the peak markers plot
            peak_markers = alt.Chart(peak_df).mark_point(
                filled=True,
                color='red',
                size=100
            ).encode(
                x=alt.X('Ramanshift', title='Raman shift/cm^-1', type='quantitative'),
                y=alt.Y('Intensity', title='Intensity/a.u.', type='quantitative'),
                tooltip=[alt.Tooltip('Ramanshift', title='Raman shift/cm^-1'),
                        alt.Tooltip('Intensity', title='Intensity/a.u.')]
            ).properties(
                width=1300,
                height=600,
                title='Spectra Average Data Plot'
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
            
            function.log_plot_generated_count(st.session_state.log_file_path)
            function.log_function_use_count(st.session_state.function_log_file_path, "Peak_Identification_Called")
        
        elif st.session_state.stats_plot_select == "Hierarchically-clustered Heatmap":
            
            st.write("**Hierarchically-clustered Heatmap**")
            
            temp = st.session_state.temp.drop(columns=['Average'])
            
            if st.session_state.HCA_heatmap:
                st.pyplot(function.hierarchical_clustering_heatmap(temp))
                function.log_function_use_count(st.session_state.function_log_file_path, "Clustermaps_Generated")
            else:
                st.pyplot(function.hierarchical_clustering_tree(temp))
                function.log_function_use_count(st.session_state.function_log_file_path, "Clustering_Dendrograms_Drawn")
        
            function.log_plot_generated_count(st.session_state.log_file_path)
        
        elif st.session_state.stats_plot_select == "Principal Components Analysis (PCA)-Beta":
            
            temp = st.session_state.temp.drop(columns=['Average'])
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
            function.log_plot_generated_count(st.session_state.log_file_path)

            st.altair_chart(function.style_altair_chart(cumulative_variance_plot), use_container_width=False)
            function.log_plot_generated_count(st.session_state.log_file_path)

            st.altair_chart(function.style_altair_chart(loading_plot), use_container_width=False)
            function.log_plot_generated_count(st.session_state.log_file_path)

            function.log_function_use_count(st.session_state.function_log_file_path, "PCA_Used")

            st.write("### PCA Scores Table")
            st.write(pca_result_df)
        
        elif st.session_state.stats_plot_select == "T-SNE Dimensionality Reduction-Beta":
            
            st.write("**T‑Distributed Stochastic Neighbor Embedding (t‑SNE) ‑ Beta**")

            temp = st.session_state.temp.drop(columns=['Average'])

            label_df = st.session_state.get('label_df')   # could be None

            tsne_df, tsne_plot = function.tsne(
                temp,
                perplexity=st.session_state.tSNE_perplexity,
                n_iter=st.session_state.tSNE_n_iter,
                label_df=label_df
            )
            st.altair_chart(function.style_altair_chart(tsne_plot), use_container_width=False)

            function.log_plot_generated_count(st.session_state.log_file_path)
            function.log_function_use_count(st.session_state.function_log_file_path, "TSNE_Used")

            st.write(tsne_df)
