import sys, json, hashlib
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.reliable.legacy_url import URLRiskEngine
from ml.reliable.legacy_sms import SMSRiskEngine
from ml.url_detection.url_features_v2 import FEATURE_NAMES, extract_features_from_series
OUT = ROOT / 'audit'
results = {}
def metric(y,p):
    pred = np.asarray(p)>=.5
    return dict(n=len(y), accuracy=accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0), f1=f1_score(y,pred,zero_division=0), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
def host(s):
    try:
        h=(urlparse(s if '://' in s else 'http://'+s).hostname or '').lower()
        return h[4:] if h.startswith('www.') else h
    except ValueError: return ''
u=URLRiskEngine(); s=SMSRiskEngine()
def upredict(values):
    output=[]
    for start in range(0,len(values),1000):
        x=pd.Series(list(values[start:start+1000]),dtype='string')
        output.extend(u.model.predict_proba(hstack([u.vectorizer.transform(x),u.scaler.transform(extract_features_from_series(x)[FEATURE_NAMES])],format='csr'))[:,1])
    return np.asarray(output)
def spredict(values):
    output=[]
    for start in range(0,len(values),1000):
        x=list(values[start:start+1000])
        output.extend(s.model.predict_proba(hstack([s.word_vectorizer.transform(x),s.char_vectorizer.transform(x)],format='csr'))[:,1])
    return np.asarray(output)
print('Reconstructing actual URL training split',flush=True)
uci=pd.read_csv(ROOT/'data/raw/phiusiil.csv',usecols=['URL','label']).drop_duplicates('URL').reset_index(drop=True)
a=pd.DataFrame({'URL':uci.URL.fillna('').astype(str),'label':(uci.label==0).astype(int)})
parts=[a]
for filename in ['benign_url_augmentation.csv','benign_deep_path_augmentation.csv']:
    d=pd.read_csv(ROOT/'data/processed'/filename,usecols=['URL']).drop_duplicates('URL').reset_index(drop=True)
    parts.append(d.assign(label=0))
combined=pd.concat(parts,ignore_index=True).drop_duplicates('URL').reset_index(drop=True)
train,test=train_test_split(combined,test_size=.2,random_state=42,stratify=combined.label)
train_urls=set(train.URL); train_hosts=set(train.URL.map(host))
domains=a.URL.map(host).drop_duplicates().to_numpy(); domains=domains[domains!='']
np.random.default_rng(42).shuffle(domains)
claimed=a[a.URL.map(host).isin(set(domains[int(len(domains)*.8):]))]
results['url_claimed_unseen_overlap']={'claimed_test_rows':len(claimed),'exact_training_urls':int(claimed.URL.isin(train_urls).sum()),'training_host_rows':int(claimed.URL.map(host).isin(train_hosts).sum())}
results['url_artifact_matches_v5']={name:hashlib.sha256((ROOT/'models'/prod).read_bytes()).hexdigest()==hashlib.sha256((ROOT/'models'/v5).read_bytes()).hexdigest() for name,prod,v5 in [('model','url_detector.joblib','url_char_logreg_v5.joblib'),('vectorizer','url_detector_vectorizer.joblib','url_char_vectorizer_v5.joblib'),('scaler','url_detector_scaler.joblib','url_char_scaler_v5.joblib')]}
print(results,flush=True)
# Evaluate real held-out UCI rows, plus subset with no training hostname overlap.
held=test[test.URL.isin(set(a.URL))].copy()
p=upredict(held.URL.tolist()); results['url_actual_uci_holdout']=metric(held.label,p)
mask=~held.URL.map(host).isin(train_hosts)
results['url_actual_unseen_hostname']=metric(held.label[mask],p[mask])
held.assign(probability=p).to_csv(OUT/'url_holdout_predictions.csv',index=False)
print(results,flush=True)
frames={k:pd.read_csv(ROOT/f'data/processed/sms/{k}.csv') for k in ['train','validation','test']}
norm=lambda x:' '.join(str(x).lower().split())
tr=set(frames['train'].text.map(norm)); va=set(frames['validation'].text.map(norm))
t=frames['test']; p=spredict(t.text.fillna('').astype(str).tolist())
results['sms_test']=metric(t.is_scam,p)
results['sms_overlap']={'test_in_train':int(t.text.map(norm).isin(tr).sum()),'test_in_validation':int(t.text.map(norm).isin(va).sum())}
mask=~t.text.map(norm).isin(tr|va)
results['sms_nonoverlapping_test']=metric(t.is_scam[mask],p[mask])
results['sms_languages']={str(lang):metric(t.is_scam[t.language==lang],p[t.language==lang]) for lang in t.language.dropna().unique()}
t.assign(probability=p).to_csv(OUT/'sms_holdout_predictions.csv',index=False)
# Fresh author-written diagnostic probes: expected labels describe scenarios, not verified live URL reputation.
cases=[]
benign_sms=[
'Your verification code is 731946. Do not share this code with anyone, including bank staff.',
'Your account statement for September is available in the banking app. No action is required.',
'Please pay the electricity bill using the official app before Friday to avoid a late fee.',
'The security team will never ask for your password or OTP. Report suspicious calls.',
'I paid for lunch yesterday. Could you send me your share when you have time?',
'Your parcel will arrive tomorrow. Track it in the app you used to place the order.',
'You have earned cashback on your purchase. It has already been credited to your wallet.',
'आज शाम छह बजे घर आ जाना, हम साथ में खाना खाएंगे।',
'आपका ओटीपी 731946 है। इसे किसी के साथ साझा न करें।',
'Kal library mein milte hain, project ki report complete karni hai.',
'Your password was changed successfully. If this was you, no further action is needed.',
'The university scholarship application is free. Apply through the student portal.'
]
scam_sms=[
'Your bank access expires in 30 minutes. Send your OTP and card PIN to this number to keep it active.',
'You were selected for a guaranteed remote job. Transfer Rs 1800 registration fee to my personal UPI before your interview.',
'Hello mum, my phone broke. This is my new number. Please urgently transfer 48000 rupees to my friend; do not call me.',
'Your parcel is held at customs. Pay Rs 17 at https://parcel-release.example/fee within one hour to prevent return.',
'I can double your savings in two days with guaranteed crypto returns. Send your deposit to my wallet now.',
'Police notice: you are under digital arrest. Transfer your savings to our safe account and do not tell anyone.',
'Your wallet requires synchronization. Reply with all twelve words of your recovery phrase.',
'We accidentally paid you too much for the item. Refund the extra using gift cards before the payment clears.',
'आपका बैंक खाता बंद हो जाएगा। इसे चालू रखने के लिए अपना ओटीपी और पिन इस नंबर पर भेजें।',
'आपने 25 लाख रुपये जीते हैं। इनाम पाने के लिए पहले 2500 रुपये की फीस भेजें।',
'Aapka account band ho jayega. OTP aur ATM PIN abhi WhatsApp par bhejo.',
'Part time review job: earn 9000 daily. First deposit 1200 to unlock your tasks and withdraw salary.'
]
for label,items in [(0,benign_sms),(1,scam_sms)]:
    for value in items: cases.append(dict(kind='sms',value=value,expected=label,scenario='author_written',seen_in_available_corpus=norm(value) in tr|va|set(t.text.map(norm))))
for value in ['https://docs.python.org/3/library/urllib.parse.html','https://github.com/scikit-learn/scikit-learn/issues/31427','https://en.wikipedia.org/wiki/Logistic_regression','https://www.rbi.org.in/Scripts/FAQView.aspx?Id=120','https://accounts.google.com/','https://www.microsoft.com/en-us/security/business','https://pypi.org/project/scikit-learn/','https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401']:
    cases.append(dict(kind='url',value=value,expected=0,scenario='benign_service_url_not_live_verified',seen_in_available_corpus=value in set(combined.URL)))
for value in ['https://paypal.com.login-confirmation.example/verify','https://sbi-kyc-renewal.example/login','https://account-restore.example/otp','https://microsoft.com@credential-check.example/signin','https://wallet-recovery.example/seed-phrase','http://paypal.com.login-confirmation.example/verify','http://sbi-kyc-renewal.example/login','http://account-restore.example/otp','http://microsoft.com@credential-check.example/signin','http://wallet-recovery.example/seed-phrase']:
    cases.append(dict(kind='url',value=value,expected=1,scenario='synthetic_phishing_structure_reserved_domain',seen_in_available_corpus=value in set(combined.URL)))
for c in cases:
    r=(u if c['kind']=='url' else s).analyze(c['value']); c.update(prediction=r['prediction'],risk_score=r['risk_score'],correct=int(r['prediction'] in ['phishing','scam'])==c['expected'])
pd.DataFrame(cases).to_csv(OUT/'fresh_probes.csv',index=False)
results['fresh_probes']={kind:dict(n=sum(c['kind']==kind for c in cases),correct=sum(c['correct'] for c in cases if c['kind']==kind),seen=sum(c['seen_in_available_corpus'] for c in cases if c['kind']==kind)) for kind in ['sms','url']}
results['fresh_failures']=[c for c in cases if not c['correct']]
results['invalid_url_probes']={v:u.analyze(v)['prediction'] for v in ['not a url','javascript:alert(1)','https://','hello world']}
(OUT/'ml_audit_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
print(json.dumps(results,indent=2,ensure_ascii=False),flush=True)
