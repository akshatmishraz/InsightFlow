import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from io import BytesIO
import hashlib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(page_title="InsightFlow", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SUPPORTED_FILES = [".csv", ".xlsx", ".xls"]


# --------------------------------------------------
# 2. FILE LOADING
# --------------------------------------------------


def load_file(file_source, file_name):
    """Load CSV or Excel files from a path or uploaded bytes."""

    extension = Path(file_name).suffix.lower()

    if isinstance(file_source, (str, Path)):

        if extension == ".csv":
            try:
                return pd.read_csv(file_source, low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(file_source, low_memory=False, encoding="latin-1")

        elif extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_source)

    else:

        if extension == ".csv":
            try:
                return pd.read_csv(BytesIO(file_source), low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(
                    BytesIO(file_source), low_memory=False, encoding="latin-1"
                )

        elif extension in [".xlsx", ".xls"]:
            return pd.read_excel(BytesIO(file_source))

    raise ValueError("Unsupported file format.")


# --------------------------------------------------
# 3. DATA PROFILING
# --------------------------------------------------


def profile_data(df):

    missing_values = df.isnull().sum()

    missing_summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": missing_values.values,
            "Missing %": (
                missing_values.values / len(df) * 100
                if len(df) > 0
                else np.zeros(len(df.columns))
            ),
        }
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": int(missing_values.sum()),
        "duplicates": int(df.duplicated().sum()),
        "empty_columns": df.columns[df.isnull().all()].tolist(),
        "missing_summary": missing_summary,
        "missing_percentage": (
            missing_values.sum() / df.size * 100 if df.size > 0 else 0
        ),
    }


# --------------------------------------------------
# 4. DATASET SESSION MANAGEMENT
# --------------------------------------------------


def activate_dataset(df, file_name, signature):
    """
    Initialize a dataset only when a new file is loaded.
    Preserve cleaning changes during normal Streamlit reruns.
    """

    current_signature = st.session_state.get("dataset_signature")

    if current_signature != signature:

        st.session_state["df"] = df.copy()

        st.session_state["cleaned_df"] = df.copy()

        st.session_state["file_name"] = file_name

        st.session_state["dataset_signature"] = signature


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 InsightFlow")

st.sidebar.caption("Your Spreadsheet Assistant")

page = st.sidebar.radio("Navigation", ["Home", "Smart Data Cleaner", "Advanced Analytics", "Business Reports"])


# --------------------------------------------------
# 6. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("📊 InsightFlow")

    st.subheader("Your Everyday Spreadsheet Assistant")

    st.write(
        "Clean spreadsheets, compare files, "
        "and generate business reports without coding."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧹 Data Cleaner")
        st.write("Detect and fix messy spreadsheets.")

    with col2:
        st.subheader("🔍 File Comparison")
        st.write("Find differences between two files.")

    with col3:
        st.subheader("📈 Report Generator")
        st.write("Generate business reports automatically.")

    st.info("Select Smart Data Cleaner from the sidebar to get started.")


# --------------------------------------------------
# 7. SMART DATA CLEANER
# --------------------------------------------------

elif page == "Smart Data Cleaner":

    st.title("🧹 Smart Data Cleaner")

    st.write(
        "Upload a spreadsheet or select a file " "already available in your workspace."
    )

    st.divider()

    input_method = st.radio(
        "Choose your file source", ["Workspace File", "Upload File"], horizontal=True
    )

    # ----------------------------------------------
    # OPTION 1: WORKSPACE FILE
    # ----------------------------------------------

    if input_method == "Workspace File":

        available_files = [
            file
            for file in DATA_DIR.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_FILES
        ]

        if available_files:

            selected_file = st.selectbox(
                "Select your spreadsheet", available_files, format_func=lambda x: x.name
            )

            if st.button("Analyze Selected File"):

                with st.spinner("Analyzing your spreadsheet..."):

                    try:

                        df = load_file(selected_file, selected_file.name)

                        file_stat = selected_file.stat()

                        signature = (
                            f"{selected_file.resolve()}_"
                            f"{file_stat.st_size}_"
                            f"{file_stat.st_mtime_ns}"
                        )

                        activate_dataset(df, selected_file.name, signature)

                        st.success(f"Successfully loaded: {selected_file.name}")

                    except Exception as error:

                        st.error(f"Unable to process file: {error}")

        else:

            st.warning(
                "No supported files found in the data folder. "
                "Please add a CSV or Excel file."
            )

    # ----------------------------------------------
    # OPTION 2: BROWSER UPLOAD
    # ----------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload your spreadsheet", type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:

            try:

                file_bytes = uploaded_file.getvalue()

                signature = hashlib.sha256(file_bytes).hexdigest()

                df = load_file(file_bytes, uploaded_file.name)

                activate_dataset(df, uploaded_file.name, signature)

                st.success(f"Successfully loaded: {uploaded_file.name}")

                st.caption(
                    f"File size: " f"{uploaded_file.size / (1024 * 1024):.2f} MB"
                )

            except Exception as error:

                st.error(f"Unable to process file: {error}")

    # ----------------------------------------------
    # DISPLAY ANALYSIS
    # ----------------------------------------------

    if "df" in st.session_state:

        df = st.session_state["df"]

        file_name = st.session_state["file_name"]

        if len(df.columns) == 0:

            st.warning("The selected file contains no columns.")

        else:

            profile = profile_data(df)

            st.divider()

            st.subheader("Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Rows", f"{profile['rows']:,}")

            col2.metric("Total Columns", profile["columns"])

            col3.metric("Missing Values", f"{profile['missing']:,}")

            col4.metric("Duplicate Rows", f"{profile['duplicates']:,}")

            st.divider()

            # --------------------------------------
            # TABS
            # --------------------------------------

            tab1, tab2 = st.tabs(
                [
                    "📋 Data Preview",
                    "🧹 Data Quality & Clean Data",
                ]
            )

            # --------------------------------------
            # TAB 1: DATA PREVIEW + COLUMN INFORMATION
            # --------------------------------------

            with tab1:

                st.subheader("Uploaded Data")

                st.dataframe(df.head(10), use_container_width=True)

                st.caption(
                    f"Showing first {min(10, len(df)):,} "
                    f"rows out of {len(df):,}."
                )

                st.divider()

                st.subheader("Column Information")

                column_info = pd.DataFrame(
                    {
                        "Column Name": df.columns,
                        "Data Type": df.dtypes.astype(str).values,
                        "Unique Values": [
                            df[col].nunique(dropna=True) for col in df.columns
                        ],
                        "Non-Null Values": df.notnull().sum().values,
                    }
                )

                st.dataframe(
                    column_info,
                    use_container_width=True,
                    hide_index=True,
                )

            # --------------------------------------
            # TAB 2: DATA QUALITY + CLEAN DATA
            # --------------------------------------

            with tab2:

                st.subheader("Data Quality Report")

                quality_col1, quality_col2 = st.columns(2)

                quality_col1.metric(
                    "Missing Cell Percentage",
                    f"{profile['missing_percentage']:.2f}%",
                )

                quality_col2.metric(
                    "Completely Empty Columns",
                    len(profile["empty_columns"]),
                )

                st.write("Missing Values by Column")

                st.dataframe(
                    profile["missing_summary"],
                    use_container_width=True,
                    hide_index=True,
                )

                if profile["empty_columns"]:

                    st.warning(
                        "Empty columns: "
                        + ", ".join(map(str, profile["empty_columns"]))
                    )

                else:

                    st.success("No completely empty columns detected.")

                if profile["duplicates"] > 0:

                    st.warning(
                        f"{profile['duplicates']:,} duplicate rows detected."
                    )

                else:

                    st.success("No duplicate rows detected.")

                st.divider()

                st.subheader("Data Cleaning Operations")

                st.write(
                    "Clean your dataset while keeping " "the original file unchanged."
                )

                original_df = st.session_state["df"]

                cleaned_df = st.session_state["cleaned_df"]

                # ----------------------------------
                # OPERATION 1: REMOVE DUPLICATES
                # ----------------------------------

                st.markdown("### 1. Remove Duplicate Rows")

                duplicate_count = int(cleaned_df.duplicated().sum())

                col1, col2 = st.columns(2)

                col1.metric("Duplicate Rows Detected", f"{duplicate_count:,}")

                col2.metric("Rows After Cleaning", f"{len(cleaned_df):,}")

                keep_option = st.radio(
                    "Which occurrence should be retained?",
                    ["Keep first occurrence", "Keep last occurrence"],
                    horizontal=True,
                )

                keep_value = (
                    "first" if keep_option == "Keep first occurrence" else "last"
                )

                rows_to_remove = cleaned_df[cleaned_df.duplicated(keep=keep_value)]

                st.write(f"**Rows that will be removed: " f"{len(rows_to_remove):,}**")

                with st.expander("Preview duplicate rows to be removed"):

                    st.dataframe(
                        rows_to_remove.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(f"Showing up to 20 of " f"{len(rows_to_remove):,} rows.")

                if st.button(
                    "Remove Duplicate Rows",
                    type="primary",
                    disabled=(duplicate_count == 0),
                ):

                    st.session_state["cleaned_df"] = cleaned_df.drop_duplicates(
                        keep=keep_value
                    ).copy()

                    st.success(
                        f"Successfully removed "
                        f"{len(rows_to_remove):,} duplicate rows!"
                    )

                    st.rerun()

                # ----------------------------------
                # OPERATION 2: HANDLE MISSING VALUES
                # ----------------------------------

                st.markdown("### 2. Handle Missing Values")

                # Always use the latest working copy
                cleaned_df = st.session_state["cleaned_df"]

                missing_operation = st.selectbox(
                    "Choose a missing-value operation",
                    ["Remove rows with missing values", "Fill missing values"],
                    key="missing_value_operation",
                )

                # ----------------------------------
                # OPTION A: REMOVE ROWS WITH MISSING VALUES
                # ----------------------------------

                if missing_operation == "Remove rows with missing values":

                    columns_to_check = st.multiselect(
                        "Check missing values in these columns",
                        options=list(cleaned_df.columns),
                        default=list(cleaned_df.columns),
                        key="missing_columns_to_check",
                    )

                    if columns_to_check:

                        rows_to_remove = cleaned_df[
                            cleaned_df[columns_to_check].isnull().any(axis=1)
                        ]

                        st.metric(
                            "Rows with Missing Values", f"{len(rows_to_remove):,}"
                        )

                        with st.expander("Preview rows that will be removed"):

                            st.dataframe(
                                rows_to_remove.head(20),
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.caption(
                                f"Showing up to 20 of "
                                f"{len(rows_to_remove):,} rows."
                            )

                        if st.button(
                            "Remove Rows with Missing Values",
                            disabled=(len(rows_to_remove) == 0),
                            key="remove_missing_rows_button",
                        ):

                            st.session_state["cleaned_df"] = cleaned_df.dropna(
                                subset=columns_to_check
                            ).copy()

                            st.success(
                                f"Removed {len(rows_to_remove):,} rows "
                                "containing missing values."
                            )

                            st.rerun()

                    else:

                        st.info("Select at least one column to check.")

                # ----------------------------------
                # OPTION B: FILL MISSING VALUES
                # ----------------------------------

                else:

                    columns_with_missing = [
                        column
                        for column in cleaned_df.columns
                        if cleaned_df[column].isnull().any()
                    ]

                    if columns_with_missing:

                        selected_column = st.selectbox(
                            "Select a column",
                            options=columns_with_missing,
                            key="fill_missing_column",
                        )

                        missing_count = int(
                            cleaned_df[selected_column].isnull().sum()
                        )

                        st.metric(
                            "Missing Values in Selected Column",
                            f"{missing_count:,}",
                        )

                        column_data = cleaned_df[selected_column]

                        is_numeric = pd.api.types.is_numeric_dtype(column_data)

                        fill_methods = (
                            ["Custom value", "Mean", "Median", "Mode"]
                            if is_numeric
                            else ["Custom value", "Mode"]
                        )

                        fill_method = st.selectbox(
                            "Choose a fill method",
                            options=fill_methods,
                            key="fill_missing_method",
                        )

                        fill_value = None

                        if fill_method == "Custom value":

                            fill_value = st.text_input(
                                "Enter the value to use", key="custom_missing_value"
                            )

                        elif fill_method == "Mean":

                            fill_value = column_data.mean()

                        elif fill_method == "Median":

                            fill_value = column_data.median()

                        elif fill_method == "Mode":

                            modes = column_data.mode(dropna=True)

                            if not modes.empty:
                                fill_value = modes.iloc[0]
                            else:
                                st.warning(
                                    "This column has no available value "
                                    "to use as its mode."
                                )

                        # Preview the result before applying it
                        if fill_value is not None:

                            try:

                                if fill_method == "Custom value" and is_numeric:
                                    fill_value = pd.to_numeric(
                                        fill_value, errors="raise"
                                    )

                                preview_df = cleaned_df.copy()

                                preview_df[selected_column] = preview_df[
                                    selected_column
                                ].fillna(fill_value)

                                st.write("Preview after filling missing values")

                                st.dataframe(
                                    preview_df[
                                        preview_df[selected_column].notnull()
                                    ].head(20),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                if st.button(
                                    "Apply Fill Operation",
                                    type="primary",
                                    key="apply_fill_missing_button",
                                ):

                                    st.session_state["cleaned_df"] = preview_df

                                    st.success(
                                        f"Filled {missing_count:,} missing values "
                                        f"in '{selected_column}'."
                                    )

                                    st.rerun()

                            except (ValueError, TypeError) as error:

                                st.error(
                                    f"Unable to apply this fill value: {error}"
                                )

                    else:

                        st.success(
                            "No missing values found in the current dataset."
                        )

                st.divider()

                # ----------------------------------
                # CLEANED DATASET
                # ----------------------------------

                st.markdown("### 3. Cleaned Dataset")

                cleaned_df = st.session_state["cleaned_df"]

                col1, col2 = st.columns(2)

                col1.metric("Original Rows", f"{len(original_df):,}")

                col2.metric("Current Rows", f"{len(cleaned_df):,}")

                st.dataframe(
                    cleaned_df.head(20), use_container_width=True, hide_index=True
                )

                # ----------------------------------
                # DOWNLOAD CLEANED DATA
                # ----------------------------------

                csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv_data,
                    file_name="InsightFlow_Cleaned_Data.csv",
                    mime="text/csv",
                )

                # ----------------------------------
                # RESET CLEANING CHANGES
                # ----------------------------------

                if st.button("Reset Cleaning Changes"):

                    st.session_state["cleaned_df"] = original_df.copy()

                    st.success("Cleaning changes have been reset.")

                    st.rerun()


# Advanced Analytics is isolated from the cleaning workflow.
if page == "Advanced Analytics":
    from advanced_analytics_ui import render_advanced_analytics
    render_advanced_analytics(load_file)


# Phase 5: standalone reporting page; analytics engines remain unchanged.
if page == "Business Reports":
    from business_reporting_ui import render_business_reports
    render_business_reports()
