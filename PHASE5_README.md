# InsightFlow — Phase 5 Business Reports

This package includes the validated application and five analytics engines plus a new deterministic report generator.

## Setup

```bash
unzip -o insightflow_phase5_business_reports.zip -d .
pip install -r requirements.txt
python -m pytest -q tests/test_advanced_analytics.py
streamlit run app.py
```

1. Load a spreadsheet in **Smart Data Cleaner**.
2. Run an analysis in **Advanced Analytics**.
3. Open **Business Reports** in the sidebar and download the HTML report.
4. Open the HTML locally and select **Print > Save as PDF** if needed.

Reports include key metrics, conservative findings, selected output tables, methodological limitations and dataset metadata. The report intentionally does not fabricate recommendations or claim causal relationships. Tables are truncated to 30 rows in the report; export full CSV data from the analysis page.

Only the latest completed analysis is available per Streamlit session. Changing the source dataset or cleaning its content invalidates the report. For dataset comparisons, the report captures the prior comparison output, but does not embed the full second dataset.

## Notes

- No paid LLM API or new dependency is required.
- The analytical engine implementations remain untouched.
- HTML is the supported report format. Browser printing can create PDF.
- Interactive browser testing and deployment remain to be done in Phase 8.
