"""Streamlit report view for the latest successfully completed analytics run."""
import streamlit as st
from business_reporting import generate_report


def render_business_reports():
    st.title('📄 Business Reports')
    st.write('Turn your latest completed analysis into a self-contained business report.')
    saved = st.session_state.get('insightflow_latest_analysis')
    if not saved:
        st.info('Run an analysis under Advanced Analytics first, then return here.')
        return
    if saved.get('dataset_signature') != st.session_state.get('dataset_signature'):
        st.warning('The active dataset changed. Run a new analysis to generate an accurate report.')
        return
    source = saved.get('source')
    if source != st.session_state.get('insightflow_analysis_source'):
        st.warning('The analysis dataset selection changed. Run a new analysis before exporting.')
        return
    df_key = 'cleaned_df' if source == 'Cleaned dataset' else 'df'
    df = st.session_state.get(df_key)
    if df is None:
        st.warning('Load a dataset before exporting a report.')
        return
    from advanced_analytics_ui import _dataset_fingerprint
    if saved.get('data_fingerprint') != _dataset_fingerprint(df):
        st.warning('The selected dataset has changed since analysis. Rerun the analysis before exporting.')
        return
    kind = saved['kind']
    st.subheader(f'Latest analysis: {kind}')
    try:
        html = generate_report(kind, saved['result'], dataset_name=st.session_state.get('file_name','Dataset'),
                               source=source, rows=len(df), columns=len(df.columns))
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f'Unable to prepare report: {exc}')
        return
    st.download_button('Download HTML report', html.encode('utf-8'),
                       file_name='InsightFlow_'+kind.lower().replace(' ','_')+'_report.html',
                       mime='text/html', type='primary')
    st.info('Open the HTML in a browser, then use Print → Save as PDF for a PDF copy. No extra dependencies required.')
    st.markdown('**Report sections:** Executive summary · Key findings · Result tables · Suggested next checks · Methodology & limitations')
