import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services.url_risk_engine import URLRiskEngine
from ml.url_detection.url_features_v2 import FEATURE_NAMES,extract_features_from_series
u=URLRiskEngine()
rows=[]
for base in ['https://docs.python.org','https://github.com','https://en.wikipedia.org']:
    for suffix in ['/','/docs/','/docs/reference/','/3/library/urllib.parse.html']:
        value=base+suffix;r=u.analyze(value)
        rows.append({'url':value,'risk_score':r['risk_score'],'prediction':r['prediction']})
x=extract_features_from_series(pd.Series(['https://docs.python.org/3/library/urllib.parse.html']))[FEATURE_NAMES]
z=u.scaler.transform(x)[0];weights=u.model.coef_[0][-len(FEATURE_NAMES):]
contrib=sorted([{'feature':f,'value':float(x.iloc[0][f]),'logit_contribution':float(z[i]*weights[i])} for i,f in enumerate(FEATURE_NAMES)],key=lambda d:abs(d['logit_contribution']),reverse=True)
data=pd.read_csv(ROOT/'data/raw/phiusiil.csv',usecols=['URL','label'])
from urllib.parse import urlparse
for label in [0,1]:
    paths=data.loc[data.label==label,'URL'].map(lambda x:urlparse(str(x)).path)
    print('UCI label',label,'rows',len(paths),'empty_or_root_path_fraction',paths.isin(['','/']).mean(),'path_length_median',paths.str.len().median())
result={'path_sensitivity':rows,'python_url_engineered_contributions':contrib[:10]}
(ROOT/'audit/path_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
