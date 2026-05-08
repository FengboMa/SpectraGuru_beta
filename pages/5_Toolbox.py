import streamlit as st
import altair as alt
import pandas as pd
from pathlib import Path
import function
import log_utils as log
import datetime as dt

function.wide_space_default()

DEFAULT_X_AXIS_TITLE = "Raman shift/cm⁻¹"
DEFAULT_Y_AXIS_TITLE = "Intensity/a.u."
PEAK_ASSIGNMENT_TABLE_PATH = Path(__file__).resolve().parents[1] / "element" / "peak_assignment_table.csv"
PEAK_ASSIGNMENT_PRIMARY_SECTIONS = (
    "Peak Row",
    "Vibrational Mode (of structural environment)",
)
PEAK_ASSIGNMENT_PEAK_ORDER = (
    "Peak",
    "Peak High",
    "Peak Low",
    "Peak Width",
    "Peak Unit",
)


def _reorder_peak_assignment_columns(df):
    ordered_peak_columns = []
    remaining_columns = []

    for column in df.columns:
        if column[0] == "Peak Row" and column[1] in PEAK_ASSIGNMENT_PEAK_ORDER:
            ordered_peak_columns.append(column)
        else:
            remaining_columns.append(column)

    ordered_peak_columns.sort(key=lambda column: PEAK_ASSIGNMENT_PEAK_ORDER.index(column[1]))
    primary_columns = [
        column
        for column in remaining_columns
        if column[0] in PEAK_ASSIGNMENT_PRIMARY_SECTIONS
    ]
    other_columns = [
        column
        for column in remaining_columns
        if column[0] not in PEAK_ASSIGNMENT_PRIMARY_SECTIONS
    ]

    return df.loc[:, ordered_peak_columns + primary_columns + other_columns]


def _flatten_peak_assignment_columns(df):
    df = df.copy()
    df.columns = [
        field if section == "Section" else f"{section} - {field}"
        for section, field in df.columns
    ]
    return df


def _build_peak_assignment_descriptions(description_rows):
    rows = []

    for section, field, description in zip(
        description_rows.iloc[0],
        description_rows.iloc[1],
        description_rows.iloc[2],
    ):
        if not any([section, field, description]):
            continue
        rows.append(
            {
                "Section": section,
                "Field": field,
                "Description": description,
            }
        )

    return pd.DataFrame(rows)


def _render_peak_assignment_info_table():
    info_table = {
        ":material/folder: Project": "**SpectraGuru** - Peak assignment table collection",
        ":material/code: Repository": "[github.com/FengboMa/SpectraGuru_beta](https://github.com/FengboMa/SpectraGuru_beta)",
        ":material/license: License": ":green-badge[Apache 2.0]",
        ":material/policy: Usage terms": ":orange-badge[Research use only] No redistribution. Users must follow the restrictions of the original sources.",
        ":material/group: Maintainers": "[Zhao Nano Lab](https://www.zhao-nano-lab.com/)",
    }
    info_df = pd.DataFrame.from_dict(info_table, orient="index", columns=["Value"])
    info_styler = info_df.style.hide(axis="columns")

    try:
        st.table(
            info_styler,
            border="horizontal",
            width="content",
        )
    except TypeError:
        st.table(info_styler)


@st.cache_data
def load_peak_assignment_table(file_path):
    description_rows = pd.read_csv(file_path, header=None, nrows=3, dtype=str, keep_default_na=False)
    peak_assignment_df = pd.read_csv(file_path, header=[0, 1], skiprows=[2])
    peak_assignment_df = _reorder_peak_assignment_columns(peak_assignment_df)
    peak_assignment_df = _flatten_peak_assignment_columns(peak_assignment_df)
    peak_assignment_df.index = pd.RangeIndex(start=1, stop=len(peak_assignment_df) + 1, name="Index")
    description_df = _build_peak_assignment_descriptions(description_rows)

    return peak_assignment_df, description_df

with st.sidebar:

    st.write("### SpectraGuru toolbox")

    st.selectbox('Select tool',
                        options=(
                            "Spectra Simulation",
                            "Peak Assignment Table",
                        ),
                        key="tool_select")

    if st.session_state.tool_select == "Spectra Simulation":
        # Display parameter select interface
        st.number_input("Number of spectra to generate", min_value=1, max_value=50, value=1, key="simulation_batch_size_select")
        st.number_input("Scale", min_value=0.01, max_value=10000.0, value=1.0, step=1.0, key="simulation_scale_select")

        st.selectbox("Select spectra structure",
                                options=(
                                    "Distinct",
                                    "Joint",
                                    "Consecutive"
                                ),
                                key="simulation_structure_select",
                                help="Distinct: All peaks are separated. Joint: Peaks are joined in pairs. Consecutive: Multiple peaks appear overlapping each other."
                            )

        structure = st.session_state.simulation_structure_select

        # Special parameters
        if structure == "Distinct":
            st.number_input("Average number of peaks", min_value=1, max_value=20, value=3, key="simulation_peak_number_select")
            st.number_input("Peak number variance", min_value=0, max_value=20, value=0, key="simulation_peak_number_variance_select", help="The maximum variation in the number of peaks, centered at the average number of peaks.")
            st.number_input("Separation factor", min_value=1.0, max_value=10.0, value=3.0, step=0.1, key="simulation_separation_factor_select", help="Specifies the minimum separation between peak centers. If set too high, some peaks may be forced to disobey the rule.")
        elif structure == "Joint":
            st.number_input("Average number of regions", min_value=1, max_value=10, value=2, key="simulation_region_number_select", help="A region refers to a conjoined pair of peaks.")
            st.number_input("Region number variance", min_value=0, max_value=10, value=1, key="simulation_region_number_variance_select", help="The maximum variation in the number of regions, centered at the average number of regions.")
            st.number_input("Clustering factor", min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="simulation_joint_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")
        elif structure == "Consecutive":
            st.number_input("Average peaks per region", min_value=1, max_value=10, value=6, key="simulation_average_peaks_per_region_select", help="A region refers to a conjoined series of peaks.")
            st.number_input("Per-region peak variance", min_value=0, max_value=10, value=2, key="simulation_per_region_peak_variance_select", help="The maximum variation in the number of peaks per region, centered at the average peaks per region.")
            st.number_input("Clustering factor", min_value=0.05, max_value=1.0, value=0.4, step=0.05, key="simulation_consecutive_clustering_factor_select", help="Specifies how closely clustered the peaks should be in each region, with lower values representing closer clustering.")

        # Baseline parameters
        st.toggle("Use baseline", key="simulation_use_baseline")

        use_baseline = st.session_state.simulation_use_baseline

        if use_baseline:
            st.selectbox("Baseline type",
                                    options=(
                                        "Polynomial",
                                        "Exponential",
                                        "Gaussian",
                                        "Sigmoidal"
                                    ),
                                    key="simulation_baseline_select",
                                    help="The shape of the baseline to be used. The baseline is normalized such that the plot spans from x=-1 to x=1."
                                )

            baseline_type = st.session_state.simulation_baseline_select

            if baseline_type == "Polynomial":
                st.caption("**Polynomial baseline:**")
                st.caption("B(x) = a0 + a1*(x) + a2*(x^2) + a3*(x^3) + a4*(x^4) + a5*(x^5)")

                for i in range(6):
                    st.slider(f"a{i}", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key=f"simulation_baseline_polynomial_a{i}")
            elif baseline_type == "Exponential":
                st.caption("**Exponential baseline:**")
                st.caption("B(x) = a*(e^(-b * (x - x0)^2)) + c*((x - x0)^2)")

                st.slider("Amplitude (a)", min_value=-2.0, max_value=2.0, value=1.0, key="simulation_baseline_exponential_amplitude")
                st.slider("Exponent (b)", min_value=-1.0, max_value=1.0, value=-0.25, key="simulation_baseline_exponential_exponent")
                st.slider("Quadratic term (c)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_exponential_quadratic_term")
                st.slider("Offset (x0)", min_value=-1.0, max_value=1.0, value=-0.5, key="simulation_baseline_exponential_offset")
            elif baseline_type == "Gaussian":
                st.caption("**Gaussian baseline:**")
                st.caption("B(x) = a * (e^(-(x - mu)^2 / (2 * sigma^2)))")

                st.slider("Amplitude (a)", min_value=-1.0, max_value=1.0, value=0.5, key="simulation_baseline_gaussian_amplitude")
                st.slider("Center (mu)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_gaussian_center")
                st.slider("Width (sigma)", min_value=0.1, max_value=1.0, value=0.5, key="simulation_baseline_gaussian_width")
            elif baseline_type == "Sigmoidal":
                st.caption("**Sigmoidal baseline:**")
                st.caption("B(x) = a / (1 + e^(-30 * k * (x - x0)))")

                st.slider("Amplitude (a)", min_value=-1.0, max_value=1.0, value=0.5, key="simulation_baseline_sigmoidal_amplitude")
                st.slider("Exponent (k)", min_value=-1.0, max_value=1.0, value=0.1, key="simulation_baseline_sigmoidal_exponent")
                st.slider("Offset (x0)", min_value=-1.0, max_value=1.0, value=0.0, key="simulation_baseline_sigmoidal_offset")

        st.toggle("Use noise", key="simulation_use_noise")

        use_noise = st.session_state.simulation_use_noise

        if use_noise:
            st.slider("Noise amplifier", min_value=0.1, max_value=20.0, value=3.0, step=0.01, key="simulation_noise_amplifier")

        st.button("Generate spectra", key="simulation_button", type="primary")


st.write("## Toolbox")

if st.session_state.tool_select == "Spectra Simulation":
    st.write("##### Spectra Simulation")
    if 'simulation_df' not in st.session_state and not st.session_state.simulation_button:
        st.write("Click \"Generate spectra\" to simulate random spectra.")
    else:
        if st.session_state.simulation_button:
            s_params = {} # Special parameters
            if structure == "Distinct":
                s_params["average_num_peaks"] = st.session_state.simulation_peak_number_select
                s_params["peak_num_variance"] = st.session_state.simulation_peak_number_variance_select
                s_params["separation_factor"] = st.session_state.simulation_separation_factor_select
            elif structure == "Joint":
                s_params["average_num_regions"] = st.session_state.simulation_region_number_select
                s_params["region_num_variance"] = st.session_state.simulation_region_number_variance_select
                s_params["clustering_factor"] = st.session_state.simulation_joint_clustering_factor_select
            elif structure == "Consecutive":
                s_params["average_peaks_per_region"] = st.session_state.simulation_average_peaks_per_region_select
                s_params["per_region_peak_variance"] = st.session_state.simulation_per_region_peak_variance_select
                s_params["clustering_factor"] = st.session_state.simulation_consecutive_clustering_factor_select

            b_params = {} # Baseline parameters
            if use_baseline:
                baseline_type = st.session_state.simulation_baseline_select
                if baseline_type == "Polynomial":
                    for i in range(6):
                        b_params[f"a{i}"] = st.session_state[f"simulation_baseline_polynomial_a{i}"]
                elif baseline_type == "Exponential":
                    b_params["a"] = st.session_state.simulation_baseline_exponential_amplitude
                    b_params["b"] = st.session_state.simulation_baseline_exponential_exponent
                    b_params["c"] = st.session_state.simulation_baseline_exponential_quadratic_term
                    b_params["x0"] = st.session_state.simulation_baseline_exponential_offset
                elif baseline_type == "Gaussian":
                    b_params["amp"] = st.session_state.simulation_baseline_gaussian_amplitude
                    b_params["c"] = st.session_state.simulation_baseline_gaussian_center
                    b_params["w"] = st.session_state.simulation_baseline_gaussian_width
                elif baseline_type == "Sigmoidal":
                    b_params["a"] = st.session_state.simulation_baseline_sigmoidal_amplitude
                    b_params["k"] = st.session_state.simulation_baseline_sigmoidal_exponent
                    b_params["x0"] = st.session_state.simulation_baseline_sigmoidal_offset

            # Call generating function
            num_spectra = st.session_state.simulation_batch_size_select
            scale = st.session_state.simulation_scale_select

            noise_amplifier = None
            if use_noise:
                noise_amplifier = st.session_state.simulation_noise_amplifier

            baseline_type = None
            if use_baseline:
                baseline_type = st.session_state.simulation_baseline_select

            st.session_state.simulation_df = function.generate_spectra(structure=structure,
                                                                        num_spectra=num_spectra,
                                                                        scale=scale,
                                                                        s_params=s_params,
                                                                        use_baseline=use_baseline,
                                                                        baseline_type=baseline_type,
                                                                        b_params=b_params,
                                                                        use_noise=use_noise,
                                                                        noise_amplifier=noise_amplifier)
            log.log_function_call("Toolbox_Spectra_Simulation", f_params={
                                    'structure':structure,
                                    's_params':s_params,
                                    'num_spectra':num_spectra,
                                    'scale':scale,
                                    'use_baseline':use_baseline,
                                    'baseline_type':baseline_type,
                                    'b_params':b_params,
                                    'use_noise':use_noise,
                                    'noise_amplifier':noise_amplifier
                                })

        simulation_df = st.session_state.simulation_df
        simulation_df_melted = simulation_df.melt(id_vars="Ramanshift", var_name="Sample ID", value_name="Intensity")
        #print("Melted", simulation_df_melted)
        simulated_spectra = alt.Chart(simulation_df_melted).mark_line().encode(
                            x=alt.X('Ramanshift', title=DEFAULT_X_AXIS_TITLE, type="quantitative"),
                            y=alt.Y('Intensity', title=DEFAULT_Y_AXIS_TITLE, type="quantitative"),
                            color="Sample ID:N",
                            size=alt.value(2), # Line width
                            tooltip=alt.value(None)
                        ).properties(
                            width=1300,
                            height=600,
                            title="Simulated Spectra"
                        )
        simulated_spectra = function.style_altair_chart(simulated_spectra)
        st.altair_chart(simulated_spectra, use_container_width=False)

        @st.cache_data
        def download_df(df):
            return df.to_csv(index = False).encode("utf-8")

        # st.write(stats_download_df)

        stats_download_df = download_df(simulation_df)
        current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_file_name = f"data_Simulated_{current_time}.csv"

        st.download_button(
            label="Download simulated data as CSV",
            data=stats_download_df,
            file_name=download_file_name,
            mime="text/csv",
        )
elif st.session_state.tool_select == "Peak Assignment Table":
    st.write("##### Peak Assignment Table")
    st.write("Browse a curated peak assignment table collection for research reference use. Move the cursor over the data table to access search, full-screen view, show/hide columns, and other table tools.")

    peak_assignment_df, peak_assignment_descriptions = load_peak_assignment_table(str(PEAK_ASSIGNMENT_TABLE_PATH))

    if not st.session_state.get("peak_assignment_table_logged", False):
        log.log_function_call(
            "Toolbox_Peak_Assignment_Table",
            f_params={
                "source_file": PEAK_ASSIGNMENT_TABLE_PATH.name,
                "rows": len(peak_assignment_df),
                "columns": len(peak_assignment_df.columns),
            },
        )
        st.session_state.peak_assignment_table_logged = True

    with st.expander("Column descriptions"):
        st.table(peak_assignment_descriptions)

    st.divider()
    st.write("##### Peak assignment data")
    st.dataframe(
        peak_assignment_df,
        use_container_width=True,
        height=700,
    )

    st.divider()
    st.write("##### Table information")
    _render_peak_assignment_info_table()
