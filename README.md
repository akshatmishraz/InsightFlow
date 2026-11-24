# InsightFlow

**A Python-powered spreadsheet cleaning and advanced analytics application built with Streamlit.**

InsightFlow helps users load spreadsheet data, prepare it for analysis, run statistical and machine-learning workflows, and export business-oriented reports through a browser interface.

> **Project status:** Advanced analytics and business reporting are implemented. Interactive end-to-end validation and public deployment are in progress.

## Features

| Area | Capabilities |
| --- | --- |
| Smart Data Cleaner | Load CSV and Excel files, inspect data quality, handle missing values and duplicates, export cleaned data |
| Statistical Analysis | Welch's t-test, correlation analysis, chi-square tests, and ANOVA |
| Anomaly Detection | IQR, Z-score, and Isolation Forest methods |
| Dataset Comparison | Compare datasets for schema changes, record differences, and distribution drift |
| Predictive Analytics | Regression and time-series forecasting workflows |
| Business Segmentation | K-Means and hierarchical clustering, PCA, and segment profiling |
| Business Reports | Generate deterministic HTML reports with key metrics, findings, output tables, and limitations |

## Technology stack

- **Application:** Python, Streamlit
- **Data processing:** pandas, NumPy
- **Statistics and machine learning:** SciPy, scikit-learn
- **Spreadsheet support:** openpyxl, xlrd
- **Testing:** pytest

## Getting started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/akshatmishraz/InsightFlow.git
cd InsightFlow
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit.

## Typical workflow

1. Open **Smart Data Cleaner** and load a CSV or Excel dataset.
2. Inspect its quality and optionally clean missing values or duplicates.
3. Open **Advanced Analytics**, choose the original or cleaned dataset, and run an analysis.
4. Review the resulting metrics and tables, and export available CSV results.
5. Open **Business Reports** to export a standalone HTML report for the latest completed analysis. Open the HTML in a browser and use **Print → Save as PDF** if a PDF is needed.

Reports are based on calculated results, without an external LLM API. They are descriptive and should not be interpreted as proof of causation. The report page retains the latest completed analysis per Streamlit session; changing the dataset or cleaning state invalidates the prior report.

## Project architecture

```text
app.py                        # Streamlit entry point and data cleaning
advanced_analytics_ui.py      # Advanced analytics interface
business_reporting_ui.py      # Business reporting interface
business_reporting.py         # Deterministic report generation
statistical_analysis.py       # Statistical tests
anomaly_detection.py          # Outlier and anomaly detection
dataset_comparison.py         # Dataset reconciliation and drift
predictive_analytics.py       # Regression and forecasting
business_segmentation.py      # Clustering and segmentation
tests/                        # Automated analytics tests
validate_workflows.py         # Workflow validation script
requirements.txt              # Python dependencies
```

## Tests

```bash
python -m pytest -q tests/test_advanced_analytics.py
python validate_workflows.py
```

Passing automated tests does not replace interactive testing of the Streamlit interface or verification of a deployed application.

## Limitations

- This project is intended for exploratory and business analytics, not automated high-stakes decisions.
- Statistical findings depend on data quality and the assumptions of the selected method.
- HTML is the native report export; PDF output is available through browser printing.
- The application has not yet been published as a verified live demo.

## Roadmap

- [x] Spreadsheet loading and cleaning
- [x] Five advanced analytics engines
- [x] Streamlit analytics interface
- [x] Deterministic business report generation
- [ ] Interactive end-to-end browser validation
- [ ] Demo dataset and documented business case study
- [ ] Streamlit Community Cloud deployment
- [ ] Screenshots and live demo link

## License

This project is distributed under the [MIT License](LICENSE).
