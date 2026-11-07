#!/bin/bash
pip install -q jarvis-tools scikit-learn tqdm numpy pandas

python -c "
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from jarvis.db.jsonutils import loadjson
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors

os.system('jarvis_populate_data.py --benchmark_file AI-SinglePropertyPrediction-formula_energy-ssub-test-mae --output_path=Out --json_key formula --id_tag id')

dataset_info = loadjson('Out/dataset_info.json')
df = pd.read_csv('Out/id_prop.csv', header=None, names=['formula', 'form_energy'])
df['id'] = df.index + 1
df['cfid_desc'] = df['formula'].apply(lambda f: get_chem_only_descriptors(f)[0])

train_df = df[:dataset_info['n_train']]
test_df  = df[dataset_info['n_train']:]

X_train = np.array(train_df['cfid_desc'].tolist())
y_train = train_df['form_energy'].values
X_test  = np.array(test_df['cfid_desc'].tolist())
y_test  = test_df['form_energy'].values

rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Test MAE:', mean_absolute_error(y_test, y_pred))

import zipfile, pandas as pd
results = pd.DataFrame({'id': test_df['id'].values, 'formula': test_df['formula'].values, 'prediction': y_pred, 'target': y_test})
fname = dataset_info['benchmark_file'] + '.csv'
results.to_csv(fname, index=False)
with zipfile.ZipFile(fname + '.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(fname, fname)
print('Done:', fname + '.zip')
"
