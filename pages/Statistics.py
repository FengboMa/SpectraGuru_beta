import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import altair as alt
# from streamlit_extras.chart_container import chart_container
from streamlit_extras.row import row
from datetime import datetime
import function

function.wide_space_default()
st.session_state.log_file_path = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v13\element\user_count.txt"
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)

if 'df' in st.session_state:
    st.sidebar.write("#### Statistics")

    # Side bar to select which plot and parameters
    st.sidebar.selectbox('Select Statistics Plot', 
                        options= ("Average Plot with Original Spectra", 
                                "Confidence Interval Plot",
                                "Correlation Heatmap"),
                        key="stats_plot_select")

    if st.session_state.stats_plot_select == "Average Plot with Original Spectra":
        st.sidebar.toggle(label='Show spectra you selected', value=True, key = 'stats_avg_act',help='Show or hide original selected spectra.')
        st.sidebar.toggle(label='Show Standard Deviation', value=True, key = 'stats_avg_std_act',help='Show or hide Standard Deviation.')
    elif st.session_state.stats_plot_select == "Correlation Heatmap":
        
        if st.sidebar.toggle(label='Customize Heatmap scale', value=False, key = 'heatmap_scale',help='Customize heatmap scale manually.'):
            st.sidebar.number_input(label='Heatmap scale min',min_value= -1.0, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_min',step = 0.01, value = 0.5, format="%.2f")
            st.sidebar.number_input(label='Heatmap scale max',min_value= -1.00, max_value= 1.00, placeholder='Insert a number between -1 and 1',
                                    key = 'heatmap_max',step = 0.01, value = 1.00, format="%.2f")
            
            if st.session_state.heatmap_min >= st.session_state.heatmap_max:
                st.sidebar.error('Invalid Number input.')
# st.sidebar.button(label="Plot", key='stats_plot',  type='primary')
# Stats section layout
""""""""""""
# Main Page
st.markdown("## Statistics")

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
                st.altair_chart(combined_plot, use_container_width=False)
                function.log_plot_generated_count(st.session_state.log_file_path)
            else:    
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
                label="Download Avg amd Std data as CSV",
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
            
            st.altair_chart(confidence_plot, use_container_width=False)
            function.log_plot_generated_count(st.session_state.log_file_path)
            
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
            st.altair_chart(combined, use_container_width=False)
            function.log_plot_generated_count(st.session_state.log_file_path)
            
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