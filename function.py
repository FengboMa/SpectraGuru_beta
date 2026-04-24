# Functions for the support of the Application
# Fast mode trigger
from curses.ascii import alt


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

# Shared helper for processing-page toggles
def clear_processing_toggles():
    import streamlit as st
    st.session_state.interpolation_act = False
    st.session_state.crop_act = False
    st.session_state.smoothening_act = False
    st.session_state.baselineremoval_act = False
    st.session_state.despike_act = False
    st.session_state.normalization_act = False
    st.session_state.outlierremoval_act = False

# Reset button function
def reset_processing():
    import streamlit as st
    st.session_state.df = st.session_state.backup.copy()
    clear_processing_toggles()
    st.session_state.preprocessing_log = []
    st.session_state.pop("remove_outliers_log", None)

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

def median_filter_spectra(spectra, window_size=3, padding_method='mirror'):
    from scipy.ndimage import median_filter

    try:
        window_size = int(window_size)
    except (TypeError, ValueError):
        raise ValueError("window_size must be a positive odd integer")

    if window_size <= 0:
        raise ValueError("window_size must be a positive odd integer")

    if window_size % 2 == 0:
        window_size += 1

    padding_aliases = {
        'edge': 'nearest',
        'zero': 'constant',
    }
    padding_method = padding_aliases.get(padding_method, padding_method)

    supported_padding_methods = {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
    if padding_method not in supported_padding_methods:
        raise ValueError(
            f"Unknown padding_method '{padding_method}'. "
            f"Supported values are: {sorted(supported_padding_methods)} plus aliases 'edge' and 'zero'."
        )

    return median_filter(
        input=spectra,
        size=window_size,
        mode=padding_method
    )

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

    # Clean up any incorrect keys (e.g., 'Spectra_processed' with lowercase p)
    corrected_counts = {}
    for key, value in counts.items():
        # Map old incorrect keys to correct ones
        if key == 'Spectra_processed':
            corrected_counts['Spectra_Processed'] = value
        else:
            corrected_counts[key] = value
    
    with open(log_file_path, "w") as file:
        for key, value in corrected_counts.items():
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

def confidence_interval(df, threshold, interval_method):
    """
    Unified interval computation.
    
    df: dataframe of replicate spectra (each column is a spectrum)
    threshold: 
        - CI mode: confidence level (90, 95, 99)
        - STD mode: standard deviation multiplier (1, 2, 3)
    interval_method: "Confidence Interval" or "Standard Deviation"
    
    Returns:
        mean_values, ci_upper, ci_lower
    """
    import numpy as np
    from scipy.stats import t

    # Number of replicate spectra
    n = df.shape[1]

    # Mean and standard deviation per row
    mean_values = df.mean(axis=1)
    sd_values = df.std(axis=1)

    # ---------------------------------------------------------
    # Mode 1: Confidence Interval (threshold = conf level)
    # ---------------------------------------------------------
    if interval_method == "Confidence Interval":

        conf_lvl = threshold

        # Standard error
        se_values = sd_values / np.sqrt(n)

        # Convert conf level to two-sided alpha
        alpha = 1 - conf_lvl / 100.0

        # t critical value
        t_value = t.ppf(1 - alpha / 2, df=n - 1)

        ci_upper = mean_values + t_value * se_values
        ci_lower = mean_values - t_value * se_values

        return mean_values, ci_upper, ci_lower

    # ---------------------------------------------------------
    # Mode 2: Standard Deviation envelope (threshold = SD multiplier)
    # ---------------------------------------------------------
    elif interval_method == "Standard Deviation":

        sd_mult = threshold

        ci_upper = mean_values + sd_mult * sd_values
        ci_lower = mean_values - sd_mult * sd_values

        return mean_values, ci_upper, ci_lower

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
                        x_label="Raman shift/cm⁻¹", y_label="Intensity/a.u.",
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
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

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

def k_nearest_neighbors(df, n_neighbors, test_set_size, label_df=None):
    import altair as alt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score, roc_curve, auc

    # transpose the dataframe 
    df_t = df.set_index("Ramanshift").T 
    samples_df = pd.DataFrame(index=df_t.index).reset_index()
    samples_df.columns = ["Ramanshift"]

    # prepare labels
    if label_df is not None:
        # rename first col to match 'Ramanshift' for merging
        first_col = label_df.columns[0]
        label_df_temp = label_df.rename(columns={first_col: "Ramanshift"})

        # remove '.txt' from Ramanshift
        samples_df["Ramanshift"] = samples_df["Ramanshift"].astype(str).str.replace(".txt", "", regex=False).str.strip()
        label_df_temp["Ramanshift"] = label_df_temp["Ramanshift"].astype(str).str.replace(".txt", "", regex=False).str.strip()

        # merge labels with samples_df
        samples_df = samples_df.merge(
            label_df_temp[["Ramanshift", "Label"]],
            on="Ramanshift",
            how="left"
        )
        y = samples_df["Label"].values # extract classes
    else:
        y = np.ones(len(df_t)) # if no labels, assign default class

    # display error message if there are less than two classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
            raise ValueError("The dataset must contain at least two different classes (labels) to perform KNN classification. "
                            "Please go back to the 'Data Upload' Page and ensure you have two or more labels in your dataset.")

    # data cleaning
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_t)

    # encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    all_classes = le.classes_
    display_names = [f"Class {int(name)}" for name in le.classes_]

    # train/test split
    if test_set_size == 0: # no test set
        X_train, X_test, y_train, y_test = X_std, X_std, y_encoded, y_encoded 
    else: 
        # calculate min test samples needed (one per class) 
        min_test_samples = len(unique_classes)
        requested_test_samples = int(len(y_encoded) * (test_set_size / 100))

        # ensure test size is large enough to include all classes
        if requested_test_samples < min_test_samples:
            final_test_size = min_test_samples # if user input is too small, use min required samples
        else:
            final_test_size = requested_test_samples # otherwise, use user-requested test size

        # split data while preserving class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_std, y_encoded, test_size=final_test_size, random_state=42, stratify=y_encoded
        )
    
    # ensure k is within a valid range based on training size
    safe_k = min(n_neighbors, len(X_train) - 1) # k cannot exceed available training samples
    safe_k = max(1, safe_k) # k is at least 1

    # model training and prediction
    knn = KNeighborsClassifier(n_neighbors=safe_k, metric="euclidean", weights="uniform")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # altair visualization of confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(all_classes)))
    cm_df = pd.DataFrame(cm, index=display_names, columns=display_names).stack().reset_index()
    cm_df.columns = ["Actual Label", "Predicted Label", "Count"]
    cm_df["Count"] = cm_df["Count"].astype(float) # fixes blank chart issue by standardizing data types

    # base for heatmap
    base = alt.Chart(cm_df).encode(
        x=alt.X("Predicted Label:N", title="Predicted Label"),
        y=alt.Y("Actual Label:N", title="Actual Label")
    )

    # confusion matrix chart 
    cm_chart = (
            base.mark_rect().encode(
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"))
            ) + 
            base.mark_text(baseline="middle").encode(
                text=alt.Text("Count:Q", format=".0f"),
                color=alt.condition(
                    alt.datum.Count > int(cm.max() / 2),
                    alt.value("white"), 
                    alt.value("black")
                )
            )
        ).properties(width=600, height=600, title="Confusion Matrix")

    # performance metrics report
    present_in_test = np.unique(y_test)
    actual_target_names = [display_names[i] for i in present_in_test]
    report_dict = classification_report(y_test, y_pred, target_names=actual_target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose() 
    df_report = df_report.loc[actual_target_names]

    # get prediction probabilities (for ROC analysis) and final predicted labels
    y_probs = knn.predict_proba(X_test)
    y_pred = knn.predict(X_test)

    # compute ROC curve and AUC for each class to see how confident model is when making predictions
    roc_list = []
    for i, class_name in enumerate(all_classes):
        if i in present_in_test:
            prob_col_idx = list(knn.classes_).index(i)
            fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, prob_col_idx])
            roc_auc = auc(fpr, tpr)
            roc_list.append(pd.DataFrame({
                "FPR": fpr, 
                "TPR": tpr, 
                "Class": f"{class_name} (AUC={roc_auc:.2f})" # AUC closer to 1 = better
            }))

    # concatenate all ROC dataframes
    df_roc = pd.concat(roc_list)

    # create ROC chart 
    roc_chart = alt.Chart(df_roc).mark_line().encode(
        x=alt.X("FPR:Q", title="False Positive Rate"), 
        y=alt.Y("TPR:Q", title="True Positive Rate"),
        color="Class:N"
    ).properties(title="ROC Curve Analysis", width=600, height=600)

    # add diagonal line for chance level
    line = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(x="x", y="y")

    # combine layers and display final format
    roc = (line + roc_chart).properties(title="ROC Curve (Receiver Operating Characteristic)", width=600, height=600)

    # output confusion matrix, peformance metrics, and ROC curve
    return cm_chart, df_report, roc

# Returns a DataFrame of spectra with the given structure
#   Distinct: Each peak is separated
#   Joint: Peaks are paired together
#   Consecutive: Multiple peaks overlap in a sequence
def generate_spectra(s_params, b_params, 
                     wavenumber_range=(400, 2000), 
                     resolution=1601, 
                     scale=1.0, 
                     structure="Distinct", 
                     use_baseline=False, 
                     baseline_type=None, 
                     use_noise=False, 
                     noise_amplifier=1, 
                     num_spectra=1):
    import pandas as pd
    import numpy as np
    from itertools import chain

    A_MIN, A_MAX = 5, 100 # Peak amplitude
    SIGMA_MIN, SIGMA_MAX = 10, 40 # Peak width
    BUFFER = 100 # Should be greater than SIGMA_MAX
    SIGMOIDAL_STEEPNESS = 30
    buffered_range = (wavenumber_range[0] + BUFFER, wavenumber_range[1] - BUFFER)

    # Establish data shape
    x = np.linspace(wavenumber_range[0], wavenumber_range[1], resolution)
    y = np.zeros((num_spectra, resolution))

    # Cuts off a subrange if it exceeds the allowed range
    def clip(range, allowed_range):
        return (max(range[0], allowed_range[0]), min(range[1], allowed_range[1]))

    # Adds a Gaussian peak to y
    def add_gaussian(y, a, mu, sigma):
        def gaussian(a, mu, sigma):
            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return y + gaussian(a, mu, sigma), a, mu, sigma

    # Adds a baseline to y
    def add_baseline(y, b_params, type="Polynomial"):
        # Normalize x to span [-1, 1]
        x_ = 2 * (x - (wavenumber_range[0] + wavenumber_range[1]) / 2) / (wavenumber_range[1] - wavenumber_range[0])

        def polynomial(a, b, c, d, e, f):
            return a*x_**5 + b*x_**4 + c*x_**3 + d*x_**2 + e*x_ + f
        def exponential(a, b, c, x0):
            return a * np.exp(-b * (x_ - x0)**2) + c * (x_ - x0)**2
        def gaussian_baseline(amp, c, w):
            return amp * np.exp(-((x_ - c) ** 2) / (2 * w ** 2))
        def sigmoidal(a, k, x0):
            return a / (1 + np.exp(-SIGMOIDAL_STEEPNESS * k * (x_ - x0)))
        
        # Extract parameters
        if type == "Polynomial":
            f, e, d, c, b, a = (b_params[i] for i in [f"a{i}" for i in range(6)])
            y += polynomial(a, b, c, d, e, f)
        elif type == "Exponential":
            a, b, c, x0 = (b_params[i] for i in ('a', 'b', 'c', 'x0'))
            y += exponential(a, b, c, x0)
        elif type == "Gaussian":
            amp, c, w = (b_params[i] for i in ('amp', 'c', 'w'))
            y += gaussian_baseline(amp, c, w)
        elif type == "Sigmoidal":
            a, k, x0 = (b_params[i] for i in ('a', 'k', 'x0'))
            y += sigmoidal(a, k, x0)
        
        return y

    # Adds Gaussian noise to y
    def add_noise(y, noise_amplifier=1):
        return y + np.random.normal(loc=0, scale=0.01*noise_amplifier, size=np.shape(y))

    # Returns an integer range centered at 'average' and with a span equal to 'variance'
    def random_select_range(average, variance, minimum=1, maximum=None):
        low = int(max(average - np.floor(variance / 2), minimum))
        high = int(average + np.ceil(variance / 2))
        if maximum is not None:
            high = int(min(high, maximum))
        return (low, high + 1)

    # Inserts a new entry to an array of ranges (2-tuples), sorted appropriately.
    def insert_sort_range(range_array, entry):
        index = 0
        while index < len(range_array) and range_array[index][0] < entry[0]:
            index += 1
        range_array.insert(index, entry)

    # Uniformly chooses a value within the provided range, but excludes ranges listed as 'excluded ranges'
    # The excluded ranges should fall within the general range and be sorted by the low end of the range
    def random_exclusive(bounds, excluded_ranges=None):
        if excluded_ranges is None:
            excluded_ranges = []

        # Find the valid ranges
        valid_ranges, total_valid_size, max_high = [], 0, bounds[0]
        for ex_range in chain(excluded_ranges, [(bounds[1], bounds[1])]):
            if ex_range[0] > max_high:
                valid_ranges.append((max_high, ex_range[0]))
                total_valid_size += ex_range[0] - max_high
            max_high = max(max_high, ex_range[1])

        #print("V", valid_ranges)
        
        if total_valid_size > 0:
            random_choice = np.random.uniform(0, total_valid_size)
            index, valid_range = 0, valid_ranges[0]
            valid_range_size = valid_range[1] - valid_range[0]
            while random_choice > valid_range_size and index + 1 < len(valid_ranges):
                random_choice -= valid_range_size
                index += 1
                valid_range = valid_ranges[index]
                valid_range_size = valid_range[1] - valid_range[0]
            #print(valid_range, random_choice)
            return valid_range[0] + random_choice
        
        # Else: excluded ranges cover the entire spectrum
        # Half the size of each excluded range and try again.
        reduced_excluded_ranges = []
        for ex_range in excluded_ranges:
            range_center = (ex_range[1] + ex_range[0]) / 2
            reduced_ex_range = ((range_center + ex_range[0]) / 2, (range_center + ex_range[1]) / 2)
            insert_sort_range(reduced_excluded_ranges, reduced_ex_range)
        return random_exclusive(bounds, reduced_excluded_ranges)

    # Add a region of peaks clumped together by a clustering factor.
    def add_region(y, allowed_range, seed, clustering_factor=0.5, num_peaks=2):

        excluded_ranges = []

        a, mu, sigma = np.zeros((3, num_peaks))
        y, a[0], mu[0], sigma[0] = add_gaussian(y, np.random.uniform(A_MIN, A_MAX), seed, np.random.uniform(SIGMA_MIN, SIGMA_MAX))
        excluded_ranges.append(clip((mu[0] - 2 * clustering_factor * SIGMA_MAX, mu[0] + 2 * clustering_factor * SIGMA_MAX), allowed_range))

        #print(mu)

        for i in range(1, num_peaks):
            leftmost_peak_center, rightmost_peak_center = mu[mu != 0].min(), mu[mu != 0].max()
            #print("LPC, RPC", leftmost_peak_center, rightmost_peak_center)
            peak_spawning_range = clip((leftmost_peak_center - 4 * clustering_factor * SIGMA_MAX, rightmost_peak_center + 4 * clustering_factor * SIGMA_MAX), allowed_range)
            #print("PSR", peak_spawning_range)
            
            y, a[i], mu[i], sigma[i] = add_gaussian(y, np.random.uniform(A_MIN, A_MAX), random_exclusive(peak_spawning_range, excluded_ranges), np.random.uniform(SIGMA_MIN, SIGMA_MAX))
            
            new_ex_range = clip((mu[i] - 2 * clustering_factor * SIGMA_MAX, mu[i] + 2 * clustering_factor * SIGMA_MAX), allowed_range)
            # Sort new excluded range by insertion
            insert_sort_range(excluded_ranges, new_ex_range)
            
        return y, a, mu, sigma

    if structure == "Distinct":
        # Extract special parameters
        average_num_peaks = s_params['average_num_peaks']
        peak_num_variance = s_params['peak_num_variance']
        separation_factor = s_params['separation_factor']
        peak_number_range = random_select_range(average=average_num_peaks, variance=peak_num_variance, maximum=20)
        
        for k in range(num_spectra):
            num_peaks = np.random.randint(peak_number_range[0], peak_number_range[1])

            #print(num_peaks)

            excluded_ranges = [] # This array must remain sorted
            for i in range(num_peaks):
                y[k], a, mu, sigma = add_gaussian(y[k], np.random.uniform(A_MIN, A_MAX), random_exclusive(buffered_range, excluded_ranges), np.random.uniform(SIGMA_MIN, SIGMA_MAX))
                # Determine the range in which new peaks should not appear
                new_ex_range = clip((mu - separation_factor * SIGMA_MAX, mu + separation_factor * SIGMA_MAX), buffered_range)

                #print(mu)
                #for ex_range in excluded_ranges:
                #    if ex_range[0] < mu and ex_range[1] > mu:
                #        print("FAIL")

                # Sort the excluded range by inserting at the correct index
                insert_sort_range(excluded_ranges, new_ex_range)
                #print(excluded_ranges)
    
    elif structure == "Joint":
        # Extract special parameters
        average_num_regions = s_params['average_num_regions']
        region_num_variance = s_params['region_num_variance']
        clustering_factor = s_params['clustering_factor'] # Determines how closely the peak pairs are joined together
        region_number_range = random_select_range(average=average_num_regions, variance=region_num_variance, maximum=10)
        
        for k in range(num_spectra):
            num_regions = np.random.randint(region_number_range[0], region_number_range[1])

            excluded_ranges = []
            for i in range(num_regions):
                y[k], a, mu, sigma = add_region(y[k], buffered_range, random_exclusive(buffered_range, excluded_ranges), clustering_factor=clustering_factor)
                region_center = np.average(mu)

                new_ex_range = clip((region_center - 16 * clustering_factor * SIGMA_MAX, region_center + 16 * clustering_factor * SIGMA_MAX), buffered_range)
                # Sort the excluded range by inserting at the correct index
                insert_sort_range(excluded_ranges, new_ex_range)

    elif structure == "Consecutive":
        # Extract special parameters
        average_peaks_per_region = s_params['average_peaks_per_region']
        per_region_peak_variance = s_params['per_region_peak_variance']
        clustering_factor = s_params['clustering_factor'] # Determines how closely the peak pairs are joined together
        region_number_range = random_select_range(average=2, variance=1)
        peak_number_range = random_select_range(average=average_peaks_per_region, variance=per_region_peak_variance, maximum=10)

        for k in range(num_spectra):
            num_regions = np.random.randint(region_number_range[0], region_number_range[1])

            excluded_ranges = []
            for i in range(num_regions):
                num_peaks = np.random.randint(peak_number_range[0], peak_number_range[1])
                y[k], a, mu, sigma = add_region(y[k], buffered_range, random_exclusive(buffered_range, excluded_ranges), clustering_factor=clustering_factor, num_peaks=num_peaks)
                region_center = np.average(mu)

                new_ex_range = clip((region_center - 30 * clustering_factor * SIGMA_MAX, region_center + 30 * clustering_factor * SIGMA_MAX), buffered_range)
                # Sort the excluded range by inserting at the correct index
                insert_sort_range(excluded_ranges, new_ex_range)
    else:
        raise ValueError(f"Unknown spectra structure: {structure}")
    
    # Normalize y
    y -= y.min()
    y /= y.max()

    if use_baseline:
        y = add_baseline(y, b_params, baseline_type)
    
    if use_noise:
        y = add_noise(y, noise_amplifier)
    
    # Renormalize
    y -= y.min()
    y /= y.max()
    y *= scale

    data = pd.DataFrame({
        "Ramanshift": x,
        **{f"y{k}": y[k] for k in range(num_spectra)}
    })

    #print(data)

    return data

# ── SNIP baseline correction ──────────────────────────────────────────────────

def lls_transform(y):
    """Log-Log-Square root transform"""
    import numpy as np
    return np.log(np.log(np.sqrt(np.maximum(y, 0) + 1) + 1) + 1)

def inv_lls_transform(v):
    """Inverse of the LLS transform."""
    import numpy as np
    return (np.exp(np.exp(v) - 1) - 1)**2 - 1

def polynomial_padding(v, pad_width, window_size=15, poly_deg=1):
    import numpy as np
    """
    Extends the array using a polynomial fit of the edges.
    
    Parameters:
    - v: The 1D array to pad.
    - pad_width: Number of points to add to each side (usually 'iterations').
    - window_size: Number of points from the edge to use for the fit.
    - poly_deg: Degree of the polynomial (1 for linear, 2 for quadratic).
    """
    n = len(v)
    # Ensure window_size isn't larger than the data
    window_size = min(window_size, n)
    
    # Left Edge
    x_left_fit = np.arange(window_size)
    y_left_fit = v[:window_size]
    coeffs_left = np.polyfit(x_left_fit, y_left_fit, poly_deg)
    
    x_left_pad = np.arange(-pad_width, 0)
    left_extension = np.polyval(coeffs_left, x_left_pad)
    
    # Right Edge
    x_right_fit = np.arange(n - window_size, n)
    y_right_fit = v[-window_size:]
    coeffs_right = np.polyfit(x_right_fit, y_right_fit, poly_deg)
    
    x_right_pad = np.arange(n, n + pad_width)
    right_extension = np.polyval(coeffs_right, x_right_pad)
    
    return np.concatenate([left_extension, v, right_extension])

def snip_1d(y, iterations=50, use_lls=True, poly_window=15, poly_deg=1, return_baseline=False):
    import numpy as np
    
    """SNIP baseline correction with polynomial edge padding and optional LLS transform."""
    
    # Preprocessing: LLS Transform
    v = lls_transform(y) if use_lls else y.astype(np.float64)
    n_original = len(v)
    
    # Padding: Polynomial Fit
    v_padded = polynomial_padding(v, iterations, window_size=poly_window, poly_deg=poly_deg)
    n_padded = len(v_padded)
    
    # Vectorized SNIP iterations
    for p in range(1, iterations + 1):
        # Center slice
        center = v_padded[p : n_padded - p]
        # Left and Right neighbors shifted by p
        left = v_padded[0 : n_padded - 2*p]
        right = v_padded[2*p : n_padded]
        
        # Apply the clipping rule
        v_padded[p : n_padded - p] = np.minimum(center, 0.5 * (left + right))
    
    # Post-processing: Remove padding and invert LLS
    v_final = v_padded[iterations : iterations + n_original]
    baseline = inv_lls_transform(v_final) if use_lls else v_final

    if return_baseline:
        return baseline
    
    return y - baseline
