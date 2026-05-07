# This file contains all manually-implemented function metadata for the purposes of displaying the function usage table and relating
# functions to their respective references/documentation.

# To add a new function to the names dict, first determine its keyname
#   Keynames should take the form DOMAIN_FEATURE_ALGORITHM with underscores used as spaces
#     - for example: Processing_Baseline_AirPLS
#     - DOMAIN should almost always be Processing, Analytics, or Toolbox
#   The pair of strings after each entry represent user-readable identifiers (FEATURE, ALGORITHM)
#     - FEATURE and ALGORITHM may be the same if appropriate
#   In the Processing/Analytics page, make sure log.log_function_call(keyname, parameters) is invoked appropriately when your
#   algorithm is called. This increments the function usage counter for your algorithm. Use the keyname from this list.
names = {
    'Processing_Despike_Auto':("Despike", "Automatic despike"),
    'Processing_Despike_Manual':("Despike", "Manual despike"),
    'Processing_Smoothing_Savgol_Filter':("Smoothing", "Savitzky-Golay filter"),
    'Processing_Smoothing_FFT_Filter':("Smoothing", "1D Fast Fourier Transform filter"),
    'Processing_Smoothing_Median_Filter':("Smoothing", "Median filter"),
    'Processing_Smoothing_Wavelet_Denoising':("Smoothing", "Wavelet denoising"),
    'Processing_Baseline_AirPLS':("Baseline removal", "AirPLS"),
    'Processing_Baseline_Mod_Poly':("Baseline removal", "Modified polynomial fitting"),
    'Processing_Baseline_Gaussian_Lorentzian_Fitting':("Baseline removal", "Gaussian-Lorentzian fitting"),
    'Processing_Baseline_ALS':("Baseline removal", "Asymmetric least squares (ALS)"),
    'Processing_Normalization_Area':("Normalization", "Normalization by area"),
    'Processing_Normalization_Peak':("Normalization", "Normalization by peak"),
    'Processing_Normalization_Minmax':("Normalization", "Min-max normalization"),
    'Processing_Remove_Outliers':("Outlier removal", "Outlier removal"),
    'Analytics_Spectra_Derivation':("Spectra derivation", "Spectra derivation"),
    'Analytics_FFT':("Fast Fourier Transform (FFT) analysis", "Fast Fourier Transform (FFT) analysis"),
    'Analytics_Correlation_Heatmap':("Correlation heatmap", "Correlation heatmap"),
    'Analytics_Peak_Identification':("Peak identification", "Peak identification"),
    'Analytics_Clustering_Clustermap':("Hierarchical clustering", "Clustermap"),
    'Analytics_Clustering_Dendrogram':("Hierarchical clustering", "Dendrogram"),
    'Analytics_PCA':("PCA", "Principal Component Analysis"),
    'Analytics_TSNE':("t-SNE", "t-Distributed Stochastic Neighbor Embedding"),
    'Analytics_ML_Classification_Random_Forest':("ML classification", "Random Forest (RF)"),
    'Analytics_ML_Classification_KNN':("ML classification", "K-nearest neighbors (KNN)"),
    'Analytics_ML_Classification_SVM':("ML classification", "Support vector machine (SVM)"),
    'Toolbox_Spectra_Simulation':("Spectra simulation", "Spectra simulation"),




}

# The list of references used by SpectraGuru. When adding entries to this list, use the blank templates at the bottom of the
# dictionary. This ensures that the existing number assignments are never altered.
#
#   "text": Either a raw string of text or a link to a reference. If a link, set "link" to True.
#   "link": True if the content of "text" is a link, and False otherwise. Be sure to replace its default value of None when filling
#      an entry.
#   "doc_page": True if the link contained in "text" is a link to a page on SpectraGuru's documentation website, and False otherwise. Be
#      sure to replace its default value of None when filling an entry.
#   "notes": Can be anything (it is not used by the application), but ideally it should provide some information to other developers about
#      what this reference is for.
references = {
    0:{"text":"Self-implemented", "link":False, "doc_page":False, "notes":None},
    1:{"text":"https://doi.org/10.1038/s41592-019-0686-2", "link":True, "doc_page":False, "notes":"SciPy reference"},
    2:{"text":"https://doi.org/10.1021/ac60214a047", "link":True, "doc_page":False, "notes":"Savitzky-Golay reference"},
    3:{"text":"https://doi.org/10.1039/b922045c", "link":True, "doc_page":False, "notes":"AirPLS reference"},
    4:{"text":"https://doi.org/10.1021/acs.analchem.5c01253", "link":True, "doc_page":False, "notes":"Optimized AirPLS reference"},
    5:{"text":"https://doi.org/10.1366/000370203322554518", "link":True, "doc_page":False, "notes":"TITLE: Automated Method for Subtraction of Fluorescence from Biological Raman Spectra"},
    6:{"text":"https://doi.org/10.1016/j.bios.2022.114721", "link":True, "doc_page":False, "notes":"TITLE: Rapid and quantitative detection of respiratory viruses using surface-enhanced Raman spectroscopy and machine learning"},
    7:{"text":"https://doi.org/10.1039/D2NR01277D", "link":True, "doc_page":False, "notes":"TITLE: Differentiation and classification of bacterial endotoxins based on surface enhanced Raman scattering and advanced machine learning"},
    8:{"text":"https://doi.org/10.1080/01621459.1963.10500845", "link":True, "doc_page":False, "notes":"Hierarchical clustering"},
    9:{"text":"https://doi.org/10.48550/arXiv.1201.0490", "link":True, "doc_page":False, "notes":"Scikit-learn reference"},
    10:{"text":"https://doi.org/10.1037/h0071325", "link":True, "doc_page":False, "notes":"PCA reference"},
    11:{"text":"https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf", "link":True, "doc_page":False, "notes":"t-SNE reference"},
    12:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Despike/", "link":True, "doc_page":True, "notes":"Despike doc page"},
    13:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Smoothening/Savitzky-Golay/", "link":True, "doc_page":True, "notes":"Savitzky-Golay doc page"},
    14:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Smoothening/Fast_Fourier_Transform/", "link":True, "doc_page":True, "notes":"FFT doc page"},
    15:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Baseline_Removal/AirPLS/", "link":True, "doc_page":True, "notes":"AirPLS doc page"},
    16:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Baseline_Removal/Mod_Poly/", "link":True, "doc_page":True, "notes":"ModPoly doc page"},
    17:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Baseline_Removal/Gaussian-Lorentzian_Fitting/", "link":True, "doc_page":True, "notes":"Gaussian-Lorentzian Fitting doc page"},
    18:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Normalization/Normalization_Area", "link":True, "doc_page":True, "notes":"Area Normalization doc page"},
    19:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Normalization/Normalization_Peak", "link":True, "doc_page":True, "notes":"Peak Normalization doc page"},
    20:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Normalization/Normalization_Minmax", "link":True, "doc_page":True, "notes":"Minmax Normalization doc page"},
    21:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Processing_Page/Processing_Feature/Outlier_Removal/", "link":True, "doc_page":True, "notes":"Outlier Removal doc page"},
    22:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Average_Plot/", "link":True, "doc_page":True, "notes":"Average Plot doc page"},
    23:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Confidence_Interval_Plot/", "link":True, "doc_page":True, "notes":"Confidence Interval Plot doc page"},
    24:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Derivative_Analysis/", "link":True, "doc_page":True, "notes":"Spectra Derivation doc page"},
    25:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Correlation_Heatmap/", "link":True, "doc_page":True, "notes":"Correlation Heatmap doc page"},
    26:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Peak_Identification/", "link":True, "doc_page":True, "notes":"Peak Identification doc page"},
    27:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Gaussian_Peak_Fitting/", "link":True, "doc_page":True, "notes":"Gaussian Peak Fitting doc page"},
    28:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Clustermap/", "link":True, "doc_page":True, "notes":"Hierarchical Clustering doc page"},
    29:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/Principal_Component_Analysis/", "link":True, "doc_page":True, "notes":"PCA doc page"},
    30:{"text":"https://fengboma.github.io/docs.spectraguru/docs/Analytics_Page/Analytics_Features/T-SNE/", "link":True, "doc_page":True, "notes":"t-SNE doc page"},
    31:{"text":"https://www.unige.ch/~sardy/Papers/robustIEEE.pdf", "link":True, "doc_page":False, "notes":"Robust Wavelet Denoising (Sardy, Tseng & Bruce 2001)"},
    32:{"text":"https://doi.org/10.1093/biomet/81.3.425", "link":True, "doc_page":False, "notes":"Ideal spatial adaptation by wavelet shrinkage (Donoho & Johnstone 1994)"},
    33:{"text":"https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf", "link":True, "doc_page":False, "notes":"Eilers and Boelens (2005), Baseline Correction with Asymmetric Least Squares Smoothing"},
    34:{"text":"", "link":None, "doc_page":None, "notes":""},
    35:{"text":"", "link":None, "doc_page":None, "notes":""},
    36:{"text":"", "link":None, "doc_page":None, "notes":""},
    37:{"text":"", "link":None, "doc_page":None, "notes":""},
    38:{"text":"", "link":None, "doc_page":None, "notes":""},
    39:{"text":"", "link":None, "doc_page":None, "notes":""},
    40:{"text":"", "link":None, "doc_page":None, "notes":""},
    41:{"text":"", "link":None, "doc_page":None, "notes":""},
    42:{"text":"", "link":None, "doc_page":None, "notes":""},
    43:{"text":"", "link":None, "doc_page":None, "notes":""},
    44:{"text":"", "link":None, "doc_page":None, "notes":""},
    45:{"text":"", "link":None, "doc_page":None, "notes":""},
    46:{"text":"", "link":None, "doc_page":None, "notes":""},
    47:{"text":"", "link":None, "doc_page":None, "notes":""},
    48:{"text":"", "link":None, "doc_page":None, "notes":""},
    49:{"text":"", "link":None, "doc_page":None, "notes":""},
    50:{"text":"", "link":None, "doc_page":None, "notes":""},
}

# This dictionary maps functions to a specific set of references from the references list. These references will be listed alongside
# the algorithms featured in the function usage table.
#
#   - Use the same keynames from the names dict.
#   - Entries are ordered arrays of integers corresponding to rows of the references dict (see above)
#   - 0 corresponds to "self-implemented." Make sure 0 is the first element of the array if it is included at all.
#   - Try not to map functions to more than 5 references unless necessary; this could clutter the function usage table.
#
#  Example: 'Processing_Normalization_Area':[0, 18] means that Area Normalization is self-implemented and has a
#  corresponding reference at references[18], which is a link to the documentation page for Area Normalization on SpectraGuru's
#  documentation website.
reference_map = {
    'Processing_Despike_Auto':[0,12],
    'Processing_Despike_Manual':[0,12],
    'Processing_Smoothing_Savgol_Filter':[1,2,13],
    'Processing_Smoothing_FFT_Filter':[0,14],
    'Processing_Smoothing_Median_Filter':[1],
    'Processing_Smoothing_Wavelet_Denoising':[31,32],
    'Processing_Baseline_AirPLS':[3,4,15],
    'Processing_Baseline_Mod_Poly':[5,16],
    'Processing_Baseline_Gaussian_Lorentzian_Fitting':[6,7,17],
    'Processing_Baseline_ALS':[0,33],
    'Processing_Normalization_Area':[0,18],
    'Processing_Normalization_Peak':[0,19],
    'Processing_Normalization_Minmax':[0,20],
    'Processing_Remove_Outliers':[0,21],
    'Analytics_Spectra_Derivation':[0,24],
    'Analytics_FFT':[0,14],
    'Analytics_Correlation_Heatmap':[0,25],
    'Analytics_Peak_Identification':[1,26],
    'Analytics_Clustering_Clustermap':[1,8,28],
    'Analytics_Clustering_Dendrogram':[1,8,28],
    'Analytics_PCA':[9,10,29],
    'Analytics_TSNE':[9,11,30],
    'Analytics_ML_Classification_Random_Forest':[0,9],
    'Analytics_ML_Classification_KNN':[0,9],
    'Analytics_ML_Classification_SVM':[0,9],
    'Toolbox_Spectra_Simulation':[0]





}
