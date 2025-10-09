# Functions for the support of the Application
# Fast mode trigger
def update_mode_option():
    import streamlit as st
    if st.session_state.backup.shape[1]>20:
        st.session_state['update_mode_option'] = True
    else:
        st.session_state['update_mode_option'] = False
        
# Default to wide
def wide_space_default():
    import streamlit as st
    st.set_page_config(layout="wide", 
                    page_icon=r"element/tab_bar_pic.png")

# Reset button function
def reset_processing():
    import streamlit as st
    st.session_state.df = st.session_state.backup.copy()
    st.session_state.interpolation_act = False
    st.session_state.crop_act = False
    st.session_state.smoothening_act = False
    st.session_state.baselineremoval_act = False
    st.session_state.despike_act = False
    st.session_state.normalization_act = False
    st.session_state.outlierremoval_act = False

# airPLS function
'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares

This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls
Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

Description from the original documentation:

Baseline drift always blurs or even swamps signals and deteriorates analytical results, particularly in multivariate analysis.  It is necessary to correct baseline drift to perform further data analysis. Simple or modified polynomial fitting has been found to be effective in some extent. However, this method requires user intervention and prone to variability especially in low signal-to-noise ratio environments. The proposed adaptive iteratively reweighted Penalized Least Squares (airPLS) algorithm doesn't require any user intervention and prior information, such as detected peaks. It iteratively changes weights of sum squares errors (SSE) between the fitted baseline and original signals, and the weights of SSE are obtained adaptively using between previously fitted baseline and original signals. This baseline estimator is general, fast and flexible in fitting baseline.


LICENCE
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''
def WhittakerSmooth(x,w,lambda_,differences=1):
    import numpy as np
    from scipy.sparse import csc_matrix, eye, diags
    from scipy.sparse.linalg import spsolve
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15, tau = 0.001):
    import numpy as np
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<tau*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

# Normalization functions
# Normalize by area
def normalize_by_area(spectra, ramanshift):
    import numpy as np
    area = np.trapz(y = spectra, x = ramanshift)
    normalized_spectra = spectra / abs(area)  # Ensure the area is always positive
    return normalized_spectra

# Normalize by peak
def normalize_by_peak(spectra):
    import numpy as np
    peak_intensity = np.max(spectra)  # Find the maximum intensity in the spectra
    normalized_spectra = spectra / peak_intensity  # Normalize by the peak intensity
    return normalized_spectra

# Min Max normalize
def min_max_normalize(spectra):
    import numpy as np
    min_val = np.min(spectra)
    max_val = np.max(spectra)
    normalized_spectra = (spectra - min_val) / (max_val - min_val)  # Normalize the single series (column)
    return normalized_spectra

# Despike
def despikeSpec(spectra, ramanshift, threshold=100, zap_length=11):
    import numpy as np
    import pandas as pd
    
    new_spectra = pd.DataFrame(np.ones_like(spectra.values), columns=spectra.columns, index=spectra.index)
    looprange = np.arange(len(ramanshift) - zap_length)
    comprange = np.arange(zap_length)

    for i in range(len(spectra.columns)):
        spec = spectra.iloc[:, i].values
        for j in looprange:
            scn = np.array([spec[j + k] for k in comprange])
            line = np.array([k * (scn[-1] - scn[0]) / zap_length + scn[0] for k in comprange])
            resid = scn - line
            for k in comprange:
                if resid[k] > threshold:
                    spec[j + k] = line[k]
        
        new_spectra.iloc[:, i] = spec

    return new_spectra

def despikeSpec_v2(spectra, ramanshift, threshold=100, zap_length=11, window_start=None, window_end=None):
    import numpy as np
    import pandas as pd

    new_spectra = pd.DataFrame(np.ones_like(spectra.values), columns=spectra.columns, index=spectra.index)
    looprange = np.arange(len(ramanshift) - zap_length)
    comprange = np.arange(zap_length)

    for i in range(len(spectra.columns)):
        spec = spectra.iloc[:, i].values
        for j in looprange:
            # Get current Raman shift at start of window
            rs_val = ramanshift.iloc[j]

            # Apply despiking only if current Raman shift is within the window
            if window_start is not None and window_end is not None:
                if not (window_start <= rs_val <= window_end):
                    continue  # Skip this point if outside the window

            scn = np.array([spec[j + k] for k in comprange])
            line = np.array([k * (scn[-1] - scn[0]) / zap_length + scn[0] for k in comprange])
            resid = scn - line
            for k in comprange:
                if resid[k] > threshold:
                    spec[j + k] = line[k]

        new_spectra.iloc[:, i] = spec

    return new_spectra
# Smoothening
def savgol_filter_spectra (spectra, window_length = 15, polyorder = 2):
    from scipy.signal import savgol_filter
    new_spectra = savgol_filter (x = spectra, 
                                window_length=window_length,
                                polyorder=polyorder)
    
    return new_spectra

# def FFT_spectra (spectra, FFT_threshold = 0.1):
#     import numpy as np
#     spectra_FFT = np.fft.fft(spectra)
    
#     threshold = int(len(spectra_FFT) * FFT_threshold)
#     spectra_FFT[threshold:-threshold] = 0
    
#     new_spectra = np.fft.ifft(spectra_FFT)
#     new_spectra = new_spectra.real
    
#     return new_spectra
def FFT_spectra(spectra, FFT_threshold=0.1, padding_method='mirror', fs=1):
    import numpy as np

    def apply_padding(signal, pad_length, method):
        if method == 'zero':
            padded_signal = np.pad(signal, (pad_length, pad_length), 'constant')
        elif method == 'mirror':
            padded_signal = np.pad(signal, (pad_length, pad_length), 'reflect')
        elif method == 'edge':
            padded_signal = np.pad(signal, (pad_length, pad_length), 'edge')
        else:
            raise ValueError("Unknown padding method")
        return padded_signal

    pad_length = len(spectra)

    # Step 1: Apply the specified padding method
    padded_signal = apply_padding(spectra, pad_length, padding_method)

    # Step 2: Apply FFT to the padded signal
    fft_result = np.fft.fft(padded_signal)
    fft_freqs = np.fft.fftfreq(len(padded_signal), 1/fs)

    # Step 3: Create a low-pass filter mask
    filter_mask = np.abs(fft_freqs) <= FFT_threshold

    # Step 4: Apply the filter mask
    fft_result_filtered = fft_result * filter_mask

    # Step 5: Apply IFFT to get the filtered signal back to the time domain
    filtered_signal_padded = np.fft.ifft(fft_result_filtered)

    # Step 6: Remove the padding
    filtered_signal = filtered_signal_padded[pad_length:2*pad_length]

    # Return the real part of the filtered signal
    return filtered_signal.real

def remove_outliers(df, single_thresh=4, distance_thresh=6, coeff_thresh=4):
    import numpy as np
    import pandas as pd
    # Separate wavenumbers and intensities
    # wavenumbers = df.iloc[:, 0]
    df = df.drop(df.columns[0], axis=1)
    intensities = df.iloc[:, 1:]
    
    # Calculate average and std spectra
    avg_spectrum = intensities.mean(axis=1)
    std_spectrum = intensities.std(axis=1)
    
    rows_to_delete = []
    deletion_reasons = {}
    
    # Process each spectrum (column)
    for col in intensities.columns:
        spectrum = intensities[col]
        
        # Single threshold rule
        if np.any(np.abs(spectrum - avg_spectrum) > single_thresh * std_spectrum):
            rows_to_delete.append(col)
            deletion_reasons[col] = "Single Threshold"
            continue
        
        # Distance rule
        distance = np.sqrt(np.sum((spectrum - avg_spectrum)**2))
        if 'distances' not in locals():
            distances = []
        distances.append(distance)
        
        if len(distances) > 1:  # We need at least 2 distances to calculate mean and std
            if distance > np.mean(distances) + distance_thresh * np.std(distances):
                rows_to_delete.append(col)
                deletion_reasons[col] = "Distance Threshold"
                continue
        
        # Coefficient rule
        correlation = np.corrcoef(spectrum, avg_spectrum)[0, 1]
        if 'correlations' not in locals():
            correlations = []
        correlations.append(correlation)
        
        if len(correlations) > 1:  # We need at least 2 correlations to calculate mean and std
            if correlation < np.mean(correlations) - coeff_thresh * np.std(correlations):
                rows_to_delete.append(col)
                deletion_reasons[col] = "Correlation Threshold"
    
    # Create a log of deleted rows
    deleted_rows_log = pd.DataFrame(
        [(col, reason) for col, reason in deletion_reasons.items()],
        columns=['Spectrum', 'Deletion Reason']
    )
    
    # Remove the outlier spectra
    df_cleaned = df.drop(columns=rows_to_delete)
    
    return df_cleaned, deleted_rows_log

# Modpoly baseline removal
def ModPoly(input_array, degree=2, repetition=100, gradient=0.001):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    '''Implementation of Modified polyfit method from paper: Automated Method for Subtraction of Fluorescence from Biological Raman Spectra, by Lieber & Mahadevan-Jansen (2003)
    
    input_array: The input data in pandas DataFrame format

    degree: Polynomial degree, default is 2

    repetition: How many iterations to run. Default is 100

    gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. If gain in any iteration is less than this, further improvement will stop
    '''

    def poly(input_array_for_poly, degree_for_poly):
        '''QR factorization of a matrix. q is orthonormal and r is upper-triangular.
        - QR decomposition is equivalent to Gram Schmidt orthogonalization, which builds a sequence of orthogonal polynomials that approximate your function with minimal least-squares error
        - Discard the first column from the resulting matrix.

        - For each value in the range of polynomial, starting from index 0 of polynomial range (for k in range(p+1)),
        create an array such that elements of array are (original_individual_value)^polynomial_index (x**k).
        - Concatenate all of these arrays created through loop as a master array using np.vstack.
        - Transpose the master array so that it's more like a tabular form using np.transpose.
        '''
        input_array_for_poly = np.array(input_array_for_poly, dtype='object')
        X = np.transpose(np.vstack([input_array_for_poly**k for k in range(degree_for_poly + 1)]))
        return np.linalg.qr(X)[0][:, 1:]

    
    yorig = input_array

    # Initial improvement criteria is set as positive infinity, to be replaced later on with actual value
    criteria = np.inf

    ywork = yorig.copy()
    yold = yorig.copy()

    polx = poly(list(range(1, len(yorig) + 1)), degree)
    nrep = 0
    lin = LinearRegression()

    while (criteria >= gradient) and (nrep <= repetition):
        ypred = lin.fit(polx, yold).predict(polx)
        ywork = np.array(np.minimum(yorig, ypred))
        criteria = sum(np.abs((ywork - yold) / yold))
        yold = ywork
        nrep += 1

    # corrected = yorig - ypred
    corrected = ypred
    corrected = np.array(list(corrected))

    return corrected

# Increments the counter for a specified metric in a given log file. Returns the new count and 
# returns 0 if the keyname does not match any recognizable keyname in the log file.
def increment_count(log_file_path, keyname, amount=1):
    try:
        counts = read_counts(log_file_path)
        counts[keyname] += amount
        write_counts(log_file_path, counts)
        return counts[keyname]
    except:
        return 0

# Returns a dictionary of all the key-value pairs expressed in a given log file. Log files must
# take the form:
#
# Key_1[\t]Value_1
# Key_2[\t]Value_2
# ...
def read_counts(log_file_path):
    import os

    counts = {}

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts = {**counts, key: int(value)}
    
    return counts

# Writes a counts dictionary to a log file
def write_counts(log_file_path, counts):
    import os

    with open(log_file_path, "w") as file:
        for key, value in counts.items():
            file.write(f"{key}\t{value}\n")


# User count function
def log_user_count(log_file_path):
    return increment_count(log_file_path, 'User')

# Plot_Generated count function
def log_plot_generated_count(log_file_path):
    return increment_count(log_file_path, 'Plot_Generated')

# Spectra_Processed count function
def log_spectra_processed_count(log_file_path):
    import streamlit as st
    return increment_count(log_file_path, 'Spectra_Processed', st.session_state.df[1:].shape[1])

# Essentially a function rename for clarity
def log_function_use_count(function_log_file_path, keyname, amount=1):
    return increment_count(function_log_file_path, keyname, amount)

# Peak finding function
def peak_identification(spectra, height=None, threshold=None, distance=None, 
                        prominence=None, width=None, wlen=None, 
                        rel_height=0.5, plateau_size=None):
    from scipy.signal import find_peaks
    
    peaks,properties = find_peaks(spectra, height=height, threshold=threshold, distance=distance, 
                        prominence=prominence, width=width, wlen=wlen, 
                        rel_height=rel_height, plateau_size=plateau_size)
    
    return peaks,properties

# Sample data
def get_transformed_spectrum_data():
    import psycopg2
    import pandas as pd
    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(
            dbname='SpectraGuruDB',
            user='sg_read',
            password='Aa123456',
            host='localhost',
            port='5432'
        )
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

    try:
        cur.execute("SELECT * FROM spectrum_data WHERE spectrum_id IN (1, 2, 3, 4, 5);")
        spectrum_data = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        # Convert the result into a pandas DataFrame
        spectrum_data = pd.DataFrame(spectrum_data, columns=column_names)
        # Display the DataFrame
        # print(spectrum_data)

        cur.execute("SELECT * FROM spectrum WHERE batch_id IN (1);")

        spectrum_name = cur.fetchall()

        column_names = [desc[0] for desc in cur.description]

        # Convert the result into a pandas DataFrame
        spectrum_name = pd.DataFrame(spectrum_name, columns=column_names)

        # Display the DataFrame
        # print(spectrum_name)

        query = '''
        SELECT 
            u.user_id AS user_id,
            u.name AS user_name,
            u.location AS user_location,
            u.institution AS user_institution,
            p.project_id AS project_id,
            p.start_date AS project_start_date,
            p.source AS project_source,
            db.batch_id AS batch_id,
            db.upload_date AS batch_upload_date,
            db.analyte_name AS batch_analyte_name,
            db.buffer_solution AS batch_buffer_solution,
            db.instrument_details AS batch_instrument_details,
            db.wavelength AS batch_wavelength,
            db.power AS batch_power,
            db.concentration AS batch_concentration,
            db.concentration_units AS batch_concentration_units,
            db.accumulation_time AS batch_accumulation_time,
            db.experimental_procedure AS batch_experimental_procedure,
            db.substrate_type AS batch_substrate_type,
            db.substrate_material AS batch_substrate_material,
            db.preparation_conditions AS batch_preparation_conditions,
            db.data_type AS batch_data_type,
            db.notes AS batch_notes
        FROM
            "user" u
        JOIN
            project_user pu ON u.user_id = pu.user_id
        JOIN
            "project" p ON p.project_id = pu.project_id
        JOIN
            project_batch pb ON p.project_id = pb.project_id
        JOIN
            "databatch" db ON db.batch_id = pb.batch_id
        WHERE db.batch_id = 1;
        '''

        # Execute the query
        cur.execute(query)

        # Fetch all results from the executed query
        rows = cur.fetchall()

        # Get column names from the cursor description
        column_names = [desc[0] for desc in cur.description]

        # Convert the result into a pandas DataFrame
        df = pd.DataFrame(rows, columns=column_names)

        # Display the DataFrame
        # print(df)

        cur.close()
        conn.close()

        spectrum_data_wide = spectrum_data.pivot(index='spectrum_id', columns='wavenumber', values='intensity')

        # Display the wide DataFrame
        # print(spectrum_data_wide)
        merged_df1 = pd.merge(spectrum_data_wide, spectrum_name, on='spectrum_id')
        merged_spectrum_data = pd.merge(merged_df1, df, on='batch_id')

        raman_shift_columns = []
        for col in merged_spectrum_data.columns[1:-9]:
            try:
                # Attempt to convert column name to float
                float(col)
                raman_shift_columns.append(col)
            except ValueError:
                # Skip columns that cannot be converted to float
                continue

        # Display the extracted columns to ensure correctness
        # print(raman_shift_columns)

        # Continue with transforming the dataframe using these extracted columns
        raman_shift_values = pd.to_numeric(raman_shift_columns)

        # Create a new dataframe similar to other_df, with the Raman shift values as the 'RamanShift' column
        transformed_df = pd.DataFrame(raman_shift_values, columns=['RamanShift'])

        # Add each spectrum as a new column to transformed_df using 'spectrum_name' from merged_spectrum_data
        for idx, row in merged_spectrum_data.iterrows():
            spectrum_name = row['spectrum_name']  # Use 'spectrum_name' from the merged dataframe
            transformed_df[spectrum_name] = row[raman_shift_columns].values
        
        return transformed_df.astype('float64')
    except:
        pass

def hierarchical_clustering_heatmap(df):
    """
    Function to create a hierarchical clustering heatmap on the sample columns of the input dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with Raman shifts as rows and samples as columns.
    
    Returns:
    sns.matrix.ClusterGrid: The generated clustermap plot.
    """
    import seaborn as sns 
    import numpy as np

    # Remove any unnamed index column if present, and set 'Ramanshift' as the index
    df_processed = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore').set_index('Ramanshift')
    
    # Create a clustermap with clustering on sample columns only
    clustermap = sns.clustermap(df_processed, 
                                row_cluster=False, 
                                col_cluster=True, 
                                method='ward', cmap="viridis", figsize=(10, 15))
    
    yticks = np.arange(0, len(df_processed.index), 100)
    clustermap.ax_heatmap.set_yticks(yticks)
    clustermap.ax_heatmap.set_yticklabels(df_processed.index[yticks])
    
    # clustermap.ax_heatmap.set_visible(False)
    # clustermap.data2d = np.full(df_processed.shape, np.nan)
    # clustermap.ax_heatmap.imshow(clustermap.data2d, cmap="Greys", aspect='auto')
    
    return clustermap

def hierarchical_clustering_tree(df):
    """
    Generates a hierarchical clustering dendrogram using Ward's method and returns the plot.
    
    Parameters:
    df (DataFrame): Input DataFrame with 'Ramanshift' as one of the columns.
    
    Returns:
    Figure: A Matplotlib Figure object with the dendrogram plot.
    """
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt
    
    # Process the DataFrame: drop 'Unnamed' columns and set index to 'Ramanshift'
    df_processed = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore').set_index('Ramanshift')
    
    # 1. Calculate the distance matrix (transpose to cluster columns)
    distance_matrix = sch.distance.pdist(df_processed.T)
    
    # 2. Apply hierarchical clustering with Ward's method
    linkage_matrix = sch.linkage(distance_matrix, method='ward')
    
    # 3. Create the dendrogram plot without displaying it
    fig, ax = plt.subplots(figsize=(10, 8))
    sch.dendrogram(linkage_matrix, labels=df_processed.columns, leaf_rotation=90, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward\'s Method)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Distance')
    
    # Return the figure
    return fig

# def pca1(df, horizontal_pc='PC1', vertical_pc='PC2'):
    
#     import altair as alt
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.decomposition import PCA
#     import pandas as pd
#     # Step 1: Drop non-numeric or irrelevant columns
#     df_transposed = df.set_index('Ramanshift').T

    
#     scaler = StandardScaler()
#     df_standardized = scaler.fit_transform(df_transposed)
    
#     # Step 3: Apply PCA
#     pca = PCA()
#     pca_components = pca.fit_transform(df_standardized)
#     explained_variance = pca.explained_variance_ratio_.cumsum()
    
#     # Create a DataFrame for PCA results
#     pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(pca_components.shape[1])])
#     pca_df['Ramanshift'] = df_transposed.index

#     # Step 4: Generate the Altair plots
#     # Plot 1: PC1 vs PC2
#     pc1_vs_pc2_plot = alt.Chart(pca_df).mark_circle(size=60).encode(
#         x=horizontal_pc,
#         y=vertical_pc,
#         tooltip=['Ramanshift', horizontal_pc, vertical_pc]
#     ).properties(
#         title=f'PCA: {horizontal_pc} vs {vertical_pc}',
#         width=1000,
#         height=500
#     )

#     # Plot 2: Cumulative Variance Explained

#     explained_variance_df = pd.DataFrame({
#         'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
#         'Cumulative Variance': explained_variance
#     })

#     # Convert the 'Component' column to a categorical type with the correct order
#     explained_variance_df['Component'] = pd.Categorical(
#         explained_variance_df['Component'],
#         categories=[f'PC{i+1}' for i in range(len(explained_variance))],
#         ordered=True
#     )

#     # Plot with Altair
#     cumulative_variance_plot = alt.Chart(explained_variance_df).mark_line(point=True).encode(
#         x=alt.X('Component', title='Principal Component'),
#         y=alt.Y('Cumulative Variance', title='Cumulative Variance Explained')
#     ).properties(
#         title='Cumulative Variance Explained by Principal Components',
#         width=1000,
#         height=500
#     )
#     # Plot 3: Loading Plot for PC1 and PC2
#     loadings = pca.components_[:3]
#     feature_names = df_transposed.columns  # Original feature names

#     # Create a DataFrame with the loadings
#     loading_df = pd.DataFrame({
#         'Feature': feature_names,
#         'PC1': loadings[0],
#         'PC2': loadings[1],
#         'PC3': loadings[2]
#     })
        
#     loading_df_melted = loading_df.melt(id_vars='Feature', var_name='Principal Component', value_name='Loading')
    
#     loading_plot = alt.Chart(loading_df_melted).mark_line(point=False).encode(
#         x=alt.X('Feature', title='Original Features'),
#         y=alt.Y('Loading', title='Loading Value'),
#         color='Principal Component',  # Different colors for PC1, PC2, and PC3
#     ).properties(
#         title='Loadings on Principal Components 1, 2, and 3',
#         width=1000,
#         height=500
#     )

#     # Return PCA-transformed data and the plots
#     return pca_df, pc1_vs_pc2_plot, cumulative_variance_plot, loading_plot

def pca(df, label_df=None, is_label=False, horizontal_pc='PC1', vertical_pc='PC2'):
    import altair as alt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd

    # Step 1: Drop non-numeric or irrelevant columns
    df_transposed = df.set_index('Ramanshift').T

    # Step 2: Standardize the data
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_transposed)

    # Step 3: Apply PCA
    pca = PCA()
    pca_components = pca.fit_transform(df_standardized)
    explained_variance = pca.explained_variance_ratio_.cumsum()

    # Step 4: Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(pca_components.shape[1])])
    pca_df['Ramanshift'] = df_transposed.index  # Assign sample names

    # Step 5: Merge with label_df if is_label is True
    if is_label and label_df is not None:
        pca_df = pca_df.merge(label_df, on='Ramanshift', how='left')

    # Step 6: Generate the Altair plots
    # Conditional color encoding based on is_label flag
    color_encoding = alt.Color('Label:N', title='Class Label') if is_label else alt.value('blue')

    pc1_vs_pc2_plot = alt.Chart(pca_df).mark_circle(size=60).encode(
        x=horizontal_pc,
        y=vertical_pc,
        color=color_encoding,  # Apply conditional coloring
        tooltip=['Ramanshift', horizontal_pc, vertical_pc] + (['Label'] if is_label else [])
    ).properties(
        title=f'PCA: {horizontal_pc} vs {vertical_pc}',
        width=1000,
        height=500
    )

    # Step 7: Cumulative Variance Explained Plot
    explained_variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Cumulative Variance': explained_variance
    })

    explained_variance_df['Component'] = pd.Categorical(
        explained_variance_df['Component'],
        categories=[f'PC{i+1}' for i in range(len(explained_variance))],
        ordered=True
    )

    cumulative_variance_plot = alt.Chart(explained_variance_df).mark_line(point=True).encode(
        x=alt.X('Component', title='Principal Component'),
        y=alt.Y('Cumulative Variance', title='Cumulative Variance Explained')
    ).properties(
        title='Cumulative Variance Explained by Principal Components',
        width=1000,
        height=500
    )

    # Step 8: Loading Plot for PC1, PC2, and PC3
    loadings = pca.components_[:3]
    feature_names = df_transposed.columns  # Original feature names

    loading_df = pd.DataFrame({
        'Feature': feature_names,
        'PC1': loadings[0],
        'PC2': loadings[1],
        'PC3': loadings[2]
    })

    loading_df_melted = loading_df.melt(id_vars='Feature', var_name='Principal Component', value_name='Loading')

    loading_plot = alt.Chart(loading_df_melted).mark_line(point=False).encode(
        x=alt.X('Feature', title='Original Features'),
        y=alt.Y('Loading', title='Loading Value'),
        color='Principal Component'
    ).properties(
        title='Loadings on Principal Components 1, 2, and 3',
        width=1000,
        height=500
    )

    # Return PCA-transformed data and the plots
    return pca_df, pc1_vs_pc2_plot, cumulative_variance_plot, loading_plot


def tsne(df, perplexity=5, n_iter=500, label_df=None):
    """
    df         : wide table with first column 'Ramanshift' and spectra columns
    label_df   : DataFrame with columns ['Ramanshift', 'Label', ...]
                 (spectrum names in col‑0, integer labels in 'Label')
    """
    import altair as alt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np

    random_state = 42

    # ------------------------------------------------------------------
    # 1.  Transpose and standardize (rows = spectra, cols = shift bins)
    # ------------------------------------------------------------------
    df_t = df.set_index('Ramanshift').T           # rows are spectra
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_t)

    # ------------------------------------------------------------------
    # 2.  t‑SNE
    # ------------------------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state
    )
    tsne_components = tsne.fit_transform(X_std)

    tsne_df = pd.DataFrame(
        tsne_components,
        columns=['TSNE1', 'TSNE2']
    )
    tsne_df['Ramanshift'] = df_t.index            # spectrum names

    # ------------------------------------------------------------------
    # 3.  Attach labels
    # ------------------------------------------------------------------
    if label_df is not None:
        # Ensure join key is named exactly 'Ramanshift'
        first_col = label_df.columns[0]
        if first_col != 'Ramanshift':
            label_df = label_df.rename(columns={first_col: 'Ramanshift'})

        tsne_df = tsne_df.merge(
            label_df[['Ramanshift', 'Label']],
            on='Ramanshift',
            how='left'
        )
    else:
        tsne_df['Label'] = 1

    # ------------------------------------------------------------------
    # 4.  Altair plot with color by Label
    # ------------------------------------------------------------------
    tsne_plot = (
        alt.Chart(tsne_df)
        .mark_circle(size=60)
        .encode(
            x='TSNE1',
            y='TSNE2',
            color=alt.Color('Label:N', legend=alt.Legend(title='Class Label')),
            tooltip=['Ramanshift', 'TSNE1', 'TSNE2', 'Label']
        )
        .properties(
            title='t‑SNE Visualization',
            width=1000,
            height=500
        )
    )

    return tsne_df, tsne_plot

def mixed_gauss_lorentz(x, A, v_g, sigma_g, L, v_l, sigma_l, I_0):
    '''
    A mixture of Gaussian and Lorentzian function for GLF fitting.

    input
        x: input wavenumber
        A: amplitude of the Gaussian function
        v_g: center of the Gaussian peak
        sigma_g: standard deviation of the Gaussian function
        L: area of the Lorentzian function
        v_l: center of the Lorentzian peak
        sigma_l: width of the Lorentzian peak
        I_0: “ground” level of the SERS spectrum at wavenumber x

    output
        the value of the gaussian-lorentzian function at wavenumber x
    '''
    import numpy as np
    gaussian = A * np.exp(-(x - v_g) ** 2 / (2 * sigma_g ** 2))
    lorentzian = (2 * L * sigma_l) / (4 * np.pi * ((x - v_l) ** 2) + sigma_l ** 2)
    return gaussian + lorentzian + I_0

def GLF(spectra_col, wavenumber, fitting_ranges, max_iteration=1000000, gtol=1e-5):
    """
    Fits a mixed Gaussian-Lorentzian baseline to a single spectrum column.
    
    Parameters:
        spectra_col: 1D np.array of spectral intensities
        wavenumber: 1D np.array of wavenumbers (same length as spectra_col)
        fitting_ranges: list of (start, end) tuples
        max_iteration: max function evaluations
        gtol: gradient tolerance

    Returns:
        corrected_spectrum: spectra_col - fitted_baseline
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # Collect data from fitting ranges
    x_data = []
    y_data = []
    for start, end in fitting_ranges:
        mask = (wavenumber >= start) & (wavenumber <= end)
        x_data.extend(wavenumber[mask])
        y_data.extend(spectra_col[mask])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Initial guess
    initial_guess = [1, np.mean(x_data), np.std(x_data), 1, np.mean(x_data), np.std(x_data), np.min(y_data)]

    # Fit
    popt, _ = curve_fit(mixed_gauss_lorentz, x_data, y_data, p0=initial_guess,
                        maxfev=max_iteration, method='trf', gtol=gtol)

    # Predict baseline and subtract
    baseline = mixed_gauss_lorentz(wavenumber, *popt)
    return baseline

#####
def style_altair_chart(chart):
    return chart.configure_axis(
        labelFontSize=16,
        titleFontSize=16,
        labelColor='#31333F',
        titleColor='#31333F'
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16,
        labelColor='#31333F',
        titleColor='#31333F'
    ).configure_title(
        fontSize=18,
        color='#31333F'
    )

# --------------------  DATA UPLOAD HELPER DISPATCHER  ----------------------
def get_db_connection():
    import psycopg2
    import streamlit as st
    return psycopg2.connect(
        dbname="SpectraGuruDB",
        user=st.session_state.user,
        password=st.session_state.passkey,
        host="localhost",
        port="5432"
    )

def compute_relevance(row, keywords):
    return sum(
        any(str(keyword).lower() in str(cell).lower() for cell in row)
        for keyword in keywords
    )

# Search Function
def search_database(search_term, data_type_filter="Both"):
    import pandas as pd
    # import psycopg2
    import streamlit as st
    # import numpy as np
    conn = None
    cur  = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        search_pattern = f"%{search_term}%"

        # Query 1 (Search in Raw Data)
        query1 = """
        SELECT 
            u.user_id AS user_id,
            u.name AS user_name,
            u.location AS user_location,
            u.institution AS user_institution,
            p.project_id AS project_id,
            p.project_name AS project_name,
            p.start_date AS project_start_date,
            p.source AS project_source,
            db.batch_id AS batch_id,  -- Keep consistent column names
            db.upload_date AS batch_upload_date,
            db.analyte_name AS batch_analyte_name,
            db.buffer_solution AS batch_buffer_solution,
            db.instrument_details AS batch_instrument_details,
            db.wavelength AS batch_wavelength,
            db.power AS batch_power,
            db.concentration AS batch_concentration,
            db.concentration_units AS batch_concentration_units,
            db.accumulation_time AS batch_accumulation_time,
            db.experimental_procedure AS batch_experimental_procedure,
            db.substrate_type AS batch_substrate_type,
            db.substrate_material AS batch_substrate_material,
            db.preparation_conditions AS batch_preparation_conditions,
            db.data_type AS batch_data_type,
            db.notes AS batch_notes,
            db.spectrum_count AS batch_spectrum_count
        FROM
            "user" u
        JOIN
            project_user pu ON u.user_id = pu.user_id
        JOIN
            "project" p ON p.project_id = pu.project_id
        JOIN
            project_batch pb ON p.project_id = pb.project_id
        JOIN
            "databatch" db ON db.batch_id = pb.batch_id
        WHERE 
            COALESCE(u.name, '') ILIKE %s OR 
            COALESCE(u.location, '') ILIKE %s OR 
            COALESCE(u.institution, '') ILIKE %s OR 
            COALESCE(p.project_name, '') ILIKE %s OR
            COALESCE(p.source, '') ILIKE %s OR 
            COALESCE(db.analyte_name, '') ILIKE %s OR
            COALESCE(db.buffer_solution, '') ILIKE %s OR
            COALESCE(db.instrument_details, '') ILIKE %s OR
            COALESCE(db.experimental_procedure, '') ILIKE %s OR
            COALESCE(db.substrate_type, '') ILIKE %s OR
            COALESCE(db.substrate_material, '') ILIKE %s OR
            COALESCE(db.preparation_conditions, '') ILIKE %s OR
            COALESCE(db.data_type, '') ILIKE %s OR
            COALESCE(db.notes, '') ILIKE %s;
        """

        # Query 2 (Search in Standard Data)
        query2 = """
        SELECT 
            u.user_id AS user_id,
            u.name AS user_name,
            u.location AS user_location,
            u.institution AS user_institution,
            p.project_id AS project_id,
            p.project_name AS project_name,
            p.start_date AS project_start_date,
            p.source AS project_source,
            db.batch_standard_id AS batch_id,  -- Keep column names same as Query 1
            db.upload_date AS batch_upload_date,
            db.analyte_name AS batch_analyte_name,
            db.buffer_solution AS batch_buffer_solution,
            db.instrument_details AS batch_instrument_details,
            db.wavelength AS batch_wavelength,
            db.power AS batch_power,
            db.concentration AS batch_concentration,
            db.concentration_units AS batch_concentration_units,
            db.accumulation_time AS batch_accumulation_time,
            db.experimental_procedure AS batch_experimental_procedure,
            db.substrate_type AS batch_substrate_type,
            db.substrate_material AS batch_substrate_material,
            db.preparation_conditions AS batch_preparation_conditions,
            db.data_type AS batch_data_type,
            db.notes AS batch_notes,
            db.spectrum_count AS batch_spectrum_count
        FROM
            "user" u
        JOIN
            project_user pu ON u.user_id = pu.user_id
        JOIN
            "project" p ON p.project_id = pu.project_id
        JOIN
            project_batch_standard pb ON p.project_id = pb.project_id
        JOIN
            "databatch_standard" db ON db.batch_standard_id = pb.batch_standard_id
        WHERE 
            COALESCE(u.name, '') ILIKE %s OR 
            COALESCE(u.location, '') ILIKE %s OR 
            COALESCE(u.institution, '') ILIKE %s OR 
            COALESCE(p.project_name, '') ILIKE %s OR
            COALESCE(p.source, '') ILIKE %s OR 
            COALESCE(db.analyte_name, '') ILIKE %s OR
            COALESCE(db.buffer_solution, '') ILIKE %s OR
            COALESCE(db.instrument_details, '') ILIKE %s OR
            COALESCE(db.experimental_procedure, '') ILIKE %s OR
            COALESCE(db.substrate_type, '') ILIKE %s OR
            COALESCE(db.substrate_material, '') ILIKE %s OR
            COALESCE(db.preparation_conditions, '') ILIKE %s OR
            COALESCE(db.data_type, '') ILIKE %s OR
            COALESCE(db.notes, '') ILIKE %s;
        """
        results = []

        keywords = search_term.strip().split()
        if not keywords:
            return pd.DataFrame()

        # Get full OR pattern
        like_patterns = [f"%{k}%" for k in keywords]

        # Set of results
        results = []

        for pattern in like_patterns:
            params = (pattern,) * 14

            if data_type_filter in ["Both", "Raw Data Only"]:
                cur.execute(query1, params)
                rows1 = cur.fetchall()
                columns1 = [desc[0] for desc in cur.description]
                df1 = pd.DataFrame(rows1, columns=columns1)
                results.append(df1)

            if data_type_filter in ["Both", "Standard Data Only"]:
                cur.execute(query2, params)
                rows2 = cur.fetchall()
                columns2 = [desc[0] for desc in cur.description]
                df2 = pd.DataFrame(rows2, columns=columns2)
                results.append(df2)

        if results:
            result_df = pd.concat(results, ignore_index=True).drop_duplicates()

            # Add a relevance score column
            result_df["relevance"] = result_df.apply(lambda row: compute_relevance(row, keywords), axis=1)

            # Sort by relevance descending
            result_df = result_df.sort_values(by="relevance", ascending=False)
        else:
            result_df = pd.DataFrame()

        
        return result_df

    except Exception as e:
        st.error(f"Error fetching search results: {e}")
        return pd.DataFrame()

    finally:
        # Always close what *was* successfully opened
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

# Better plot downloading
def make_matplotlib_png(data, x_col,
                        plot_width_in=8.0, legend_width_in=4.5, height_in=6.0,
                        legend_fontsize=11):
    import io
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    import streamlit as st
    """
    Return a publication-ready PNG (bytes) from `data`.
    Expects columns: [x_col, 'Intensity', 'Sample ID'].
    Style: Times New Roman, no title, inward ticks + minors, boxed axes,
    rainbow colors; 'Average' plotted last as dashed black.
    """
    # set + later restore rcParams so we don't leak styles
    old_rc = plt.rcParams.copy()
    try:
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.major.size": 6,
            "ytick.minor.size": 3,
            "legend.frameon": False,
            "savefig.dpi": 600,
        })

        # --- figure with two columns: left=plot (fixed width), right=legend (extra width)
        fig = plt.figure(figsize=(plot_width_in + legend_width_in, height_in), constrained_layout=False)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[plot_width_in, legend_width_in])
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")  # legend-only area

        # Spines & ticks: boxed spines; ticks only bottom/left
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(1.2)
        ax.tick_params(axis="both", which="both",
                       bottom=True, left=True, top=False, right=False,
                       direction="in")

        # Order samples so 'Average' is plotted last
        sids = list(dict.fromkeys(data["Sample ID"].astype(str)))
        sids = [s for s in sids if s != "Average"] + (["Average"] if "Average" in sids else [])

        # Rainbow colors for non-average lines
        n_nonavg = max(1, len([s for s in sids if s != "Average"]))
        cmap = plt.get_cmap("rainbow")
        ci = 0

        for sid in sids:
            d = data[data["Sample ID"].astype(str) == sid].sort_values(by=x_col)
            if d.empty:
                continue
            if sid == "Average":
                ax.plot(d[x_col], d["Intensity"], "--", linewidth=2.4, color="black", label=sid, zorder=3)
            else:
                ax.plot(d[x_col], d["Intensity"], "-", linewidth=1.6, alpha=0.98,
                        color=cmap(ci / (n_nonavg - 1 if n_nonavg > 1 else 1)), label=sid)
                ci += 1

        # Labels (no title)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")

        # Minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Build legend in the dedicated right panel (so it doesn't push the plot)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_leg.legend(handles, labels, loc="center left", fontsize=legend_fontsize,
                          ncol=1, handlelength=2.5, borderaxespad=0.0, frameon=False)

        # Keep margins tidy, leave small gap between plot and legend column
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98, wspace=0.05)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    finally:
        plt.rcParams.update(old_rc)

def spectra_derivation(
    group: "pd.DataFrame",
    norm_method: str = "None",
    sg_win: int = 11,   # kept for compatibility; ignored
    sg_poly: int = 3    # kept for compatibility; ignored
) -> "pd.DataFrame":
    """
    Compute 1st and 2nd derivatives of a single spectrum group (per Sample ID),
    with optional normalization applied AFTER derivatives.
    
    Parameters
    ----------
    group : pd.DataFrame
        Columns: 'Ramanshift', 'Intensity' for one Sample ID.
    norm_method : str
        "None" or "Min-Max Normalization".
    sg_win, sg_poly : int
        Ignored in this derivative-only implementation (kept for compatibility).

    Returns
    -------
    pd.DataFrame
        Copy of input (sorted by Ramanshift) with two new columns:
        'y1' (1st derivative), 'y2' (2nd derivative).
        If normalization is enabled, y1/y2 are min–max scaled per spectrum.
    """

    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter

    g = group.sort_values("Ramanshift").copy()
    x = g["Ramanshift"].to_numpy(dtype=float)
    y = g["Intensity"].to_numpy(dtype=float)

    n = len(x)
    if n < 5:
        g["y1"] = np.nan
        g["y2"] = np.nan
        return g

    # --- validate window and poly ---
    w = int(sg_win)
    if w % 2 == 0:  # must be odd
        w += 1
    if w > n:       # cannot exceed number of points
        w = n if n % 2 == 1 else n - 1
    if w < 5:
        w = 5
    p = int(min(sg_poly, w - 1))

    # --- compute spacing for delta ---
    dx = np.diff(x)
    delta = np.median(dx) if len(dx) else 1.0
    if not np.isfinite(delta) or delta == 0:
        delta = 1.0

    # --- derivatives with Savitzky–Golay ---
    y1 = savgol_filter(y, window_length=w, polyorder=p, deriv=1, delta=delta, mode="interp")
    y2 = savgol_filter(y, window_length=w, polyorder=p, deriv=2, delta=delta, mode="interp")

    # --- normalization AFTER derivatives ---
    if norm_method == "Min-Max Normalization":
        def _minmax(a: np.ndarray) -> np.ndarray:
            amin = np.nanmin(a)
            amax = np.nanmax(a)
            rng = amax - amin
            if np.isfinite(rng) and rng > 0:
                return (a - amin) / rng
            return np.zeros_like(a)

        y1 = _minmax(y1)
        y2 = _minmax(y2)

    g["y1"] = y1
    g["y2"] = y2
    return g