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
                    page_icon=r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\tab_bar_pic.png")

# Reset button function
def reset_processing():
    import streamlit as st
    st.session_state.df = st.session_state.backup
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

def airPLS(x, lambda_=100, porder=1, itermax=15):
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
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
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
def normalize_by_peak(spectra, ramanshift):
    import numpy as np
    peak_intensity = np.max(spectra)  # Find the maximum intensity in the spectra
    normalized_spectra = spectra / peak_intensity  # Normalize by the peak intensity
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

# User count function
def log_user_count(log_file_path):
    import os
    # Initialize counts
    counts = {
        "User": 0,
        "Plot_Generated": 0,
        "Spectra_processed": 0
    }
    
    # Read the current counts from the log file
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts[key] = int(value)

    # Increment the user count
    counts["User"] += 1

    # Write the new counts back to the log file
    with open(log_file_path, "w") as file:
        for key, value in counts.items():
            file.write(f"{key}\t{value}\n")
    
    return counts["User"]

def read_counts(log_file_path):
    import os
    counts = {
        "User": 0,
        "Plot_Generated": 0,
        "Spectra_processed": 0
    }
    
    # Read the current counts from the log file
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts[key] = int(value)
    
    return counts

# Plot_Generated count function
def log_plot_generated_count(log_file_path):
    import os
    # Initialize counts
    counts = {
        "User": 0,
        "Plot_Generated": 0,
        "Spectra_processed": 0
    }
    
    # Read the current counts from the log file
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts[key] = int(value)

    # Increment the user count
    counts["Plot_Generated"] += 1

    # Write the new counts back to the log file
    with open(log_file_path, "w") as file:
        for key, value in counts.items():
            file.write(f"{key}\t{value}\n")
    
    return counts["Plot_Generated"]

# Plot_Generated count function
def log_spectra_processed_count(log_file_path):
    import os
    import streamlit as st
    # Initialize counts
    counts = {
        "User": 0,
        "Plot_Generated": 0,
        "Spectra_Processed": 0
    }
    
    # Read the current counts from the log file
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split('\t')
                counts[key] = int(value)

    # Increment the user count
    counts["Spectra_Processed"] += st.session_state.df[1:].shape[1]

    # Write the new counts back to the log file
    with open(log_file_path, "w") as file:
        for key, value in counts.items():
            file.write(f"{key}\t{value}\n")
    
    return counts["Spectra_Processed"]