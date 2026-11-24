"""Smoke-test InsightFlow's five engines without launching Streamlit."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from statistical_analysis import correlation_analysis
from anomaly_detection import detect_univariate_anomalies
from dataset_comparison import compare_datasets
from predictive_analytics import regression_analysis, forecast_time_series
from business_segmentation import segment_dataset

rng = np.random.default_rng(19)
df = pd.DataFrame({'id': np.arange(60), 'x': np.arange(60, dtype=float), 'y': 3*np.arange(60, dtype=float)+5,
                   'group': ['A']*30+['B']*30})
before = df.copy(deep=True)
assert correlation_analysis(df, ['x','y'])['correlation'].loc['x','y'] > .99
assert detect_univariate_anomalies(pd.DataFrame({'v':[1,2,2,3,4,100]}), 'v')['anomaly_count'] >= 1
new = df.copy(); new.loc[0,'y'] = 999; new = pd.concat([new, pd.DataFrame([{'id':60,'x':60.,'y':185.,'group':'B'}])],ignore_index=True)
r = compare_datasets(df,new,['id'],['y']); assert len(r['added_records'])==1 and len(r['modified_records'])==1
assert regression_analysis(df,['x'],'y')['rmse'] < 1e-8
dates = pd.DataFrame({'date':pd.date_range('2025-01-01',periods=40,freq='D'),'value':np.arange(40,dtype=float)})
assert len(forecast_time_series(dates,'date','value',periods=5,frequency='D')['forecast']) == 5
assert segment_dataset(df,['x','y'],n_clusters=2)['data']['segment'].nunique()==2
assert_frame_equal(df,before)
print('PASS: all five engine workflows and source-data immutability')
