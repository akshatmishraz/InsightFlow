"""Deterministic, escaped HTML business reports for InsightFlow analytics outputs."""
from datetime import datetime, timezone
from html import escape
import math
import numpy as np
import pandas as pd


def _number(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return f'{value:,.4g}' if math.isfinite(float(value)) else 'N/A'
    return str(value)


def _table(frame, limit=30):
    if isinstance(frame, pd.Series):
        frame = frame.rename('value').reset_index()
    if not isinstance(frame, pd.DataFrame):
        return ''
    view = frame.head(limit).copy()
    view.columns = [str(c) for c in view.columns]
    return view.to_html(index=False, escape=True, border=0, classes='data-table', na_rep='—') + (f'<p class="small">Showing first {limit} rows.</p>' if len(frame)>limit else '')


def _safe(v):
    return escape(_number(v))


def _summarize(kind, result):
    metrics, findings, caveats, tables = {}, [], [], []
    if kind == 'Correlation':
        matrix = result['correlation']
        pairs = []
        for i, a in enumerate(matrix.columns):
            for b in matrix.columns[i+1:]:
                val = matrix.loc[a,b]
                if pd.notna(val): pairs.append((abs(float(val)), str(a), str(b), float(val)))
        metrics['Valid pairs'] = len(pairs)
        if pairs:
            _, a, b, val = max(pairs)
            findings.append(f'Strongest observed absolute correlation: {a} vs {b} (r={val:.3f}).')
        else: findings.append('No valid pairwise correlations were available.')
        caveats.append('Correlation does not establish causation; inspect sample counts and potential confounders.')
        tables.append(('Correlation matrix', matrix.reset_index()))
        if 'pair_counts' in result: tables.append(('Pairwise sample counts', result['pair_counts'].reset_index()))
    elif kind in ('Welch t-test', 'Chi-square', 'One-way ANOVA'):
        for key in ('p_value', 'n_a', 'n_b', 'sample_size', 'group_count', 'mean_difference_b_minus_a', 'chi2', 'f_statistic'):
            if key in result: metrics[key.replace('_',' ').title()] = result[key]
        if 'p_value' in result:
            findings.append(f'Observed p-value: {_number(result["p_value"])}; interpret against the preselected significance threshold.')
        caveats.append('Statistical significance does not imply practical significance or causation. Check assumptions and effect sizes.')
        if 'group_summary' in result: tables.append(('Group summary', result['group_summary']))
    elif kind == 'Anomaly Detection':
        for k in ('rows_scored','excluded_rows','anomaly_count','anomaly_percentage','anomaly_percentage_of_scored'):
            if k in result: metrics[k.replace('_',' ').title()] = result[k]
        findings.append(f'{result.get("anomaly_count", 0)} records flagged by {result.get("method", "the selected method")}.')
        caveats.append('Flagged records are candidates for investigation, not confirmed errors or fraud.')
        if 'data' in result:
            flagged = result['data'].loc[result['data']['is_anomaly'].fillna(False)]
            tables.append(('Flagged records', flagged.reset_index(drop=True)))
    elif kind == 'Dataset Comparison':
        for label, key in [('Added','added_records'),('Deleted','deleted_records'),('Modified','modified_records')]:
            metrics[label] = len(result.get(key, []))
        metrics['Unchanged shared'] = result.get('unchanged_shared_records', 0)
        findings.append(f'{metrics["Added"]} added, {metrics["Deleted"]} deleted and {metrics["Modified"]} modified records.')
        schema = result.get('schema_changes',{})
        findings.append(f'Added columns: {len(schema.get("added_columns", []))}; removed columns: {len(schema.get("removed_columns", []))}.')
        caveats.append('Results depend on correctly chosen unique business keys; drift tests are exploratory.')
        for key in ('added_records','deleted_records'):
            if key in result: tables.append((key.replace('_',' ').title(),result[key]))
        if result.get('numeric_distribution_drift'): tables.append(('Distribution drift',pd.DataFrame.from_dict(result['numeric_distribution_drift'],orient='index').reset_index()))
    elif kind == 'Regression':
        for k in ('train_rows','test_rows','mae','rmse','r2','mape_percent'):
            if k in result: metrics[k.replace('_',' ').upper() if k in ('mae','rmse','r2') else k.replace('_',' ').title()] = result[k]
        findings.append(f'Holdout RMSE: {_number(result.get("rmse", "N/A"))}; MAE: {_number(result.get("mae", "N/A"))}.')
        caveats.append('Holdout metrics are not a guarantee of future performance; check leakage, drift and baseline performance.')
        for key in ('predictions','coefficients'):
            if key in result: tables.append((key.title(),result[key]))
    elif kind == 'Forecast':
        for k in ('mae','rmse','frequency'):
            if k in result: metrics[k.upper()] = result[k]
        findings.append(f'Forecast generated for {len(result["forecast"])} future periods.')
        caveats.append('Forecasts are model estimates; confirm backtest horizon, seasonality and assumptions before use.')
        tables.append(('Future forecast',result['forecast']))
    elif kind == 'Business Segmentation':
        for k in ('silhouette_score','davies_bouldin_score'):
            if k in result: metrics[k.replace('_',' ').title()] = result[k]
        sizes = result.get('cluster_sizes')
        if sizes is not None:
            findings.append(f'{len(sizes)} clusters produced across {int(sizes.sum())} scored records.')
            tables.append(('Segment sizes',sizes))
        if 'profile' in result: tables.append(('Segment profiles',result['profile'].reset_index()))
        caveats.append('Clusters are exploratory groupings, not causal or validated customer personas.')
    else:
        raise ValueError(f'Unsupported report type: {kind}')
    if result.get('note'): caveats.append(str(result['note']))
    return metrics, findings, caveats, tables


def generate_report(kind, result, *, dataset_name='Dataset', source='Selected dataset', rows=None, columns=None):
    """Return a standalone UTF-8 HTML document. No AI-generated claims or external assets."""
    if not isinstance(result, dict): raise TypeError('Analysis result must be a dictionary.')
    metrics, findings, caveats, tables = _summarize(kind, result)
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    meta = [('Dataset',dataset_name),('Source',source),('Generated',generated)]
    if rows is not None: meta.append(('Dataset rows',rows))
    if columns is not None: meta.append(('Dataset columns',columns))
    def lis(items): return ''.join('<li>'+escape(str(x))+'</li>' for x in items)
    metric_html = ''.join(f'<div class="metric"><small>{_safe(k)}</small><strong>{_safe(v)}</strong></div>' for k,v in metrics.items())
    table_html = ''.join(f'<section><h2>{_safe(title)}</h2>{_table(frame)}</section>' for title,frame in tables)
    meta_html = ' · '.join(f'{_safe(k)}: {_safe(v)}' for k,v in meta)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>InsightFlow — {_safe(kind)} Report</title><style>
    body{{font:15px/1.55 system-ui,Arial,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;color:#17263a}}
    h1{{font-size:30px;margin-bottom:6px}}h2{{font-size:19px;margin-top:30px}}.small,small{{color:#607086;font-size:12px}}
    .metrics{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}.metric{{padding:14px 18px;border:1px solid #d9e1e9;border-radius:8px;min-width:135px}}
    .metric strong{{display:block;font-size:21px}}.data-table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}
    th,td{{padding:8px 10px;border-bottom:1px solid #e2e7ed;text-align:left;white-space:nowrap}}th{{background:#f4f7fa}}
    section{{break-inside:avoid}}@media print{{body{{margin:15mm}}.data-table{{display:table}}}}
    </style></head><body><h1>InsightFlow · {_safe(kind)}</h1><p class="small">{meta_html}</p>
    <h2>Executive summary</h2><p>This report summarizes the recorded {_safe(kind).lower()} analysis of the selected dataset. All findings are based on the supplied computed outputs.</p>
    <div class="metrics">{metric_html}</div><section><h2>Key findings</h2><ul>{lis(findings)}</ul></section>
    <section><h2>Suggested next checks</h2><p>Validate the findings against business definitions, source quality and relevant decision thresholds before taking action.</p></section>
    {table_html}<section><h2>Methodology & limitations</h2><ul>{lis(caveats)}</ul></section>
    <p class="small">Generated automatically by InsightFlow. Tables are limited to 30 rows for readability; use the analytics CSV exports for full detail.</p></body></html>'''
