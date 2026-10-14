import hashlib,json,urllib.request,urllib.error
from pathlib import Path
from sqlalchemy import select
from backend.app.database.database import SessionLocal
from backend.app.models.analysis import Analysis
from backend.app.services.threat_analysis_service import ThreatAnalysisService
from ml.reliable.prepare import ROOT,sha
service=ThreatAnalysisService();results=[]
for kind,value in [('url','github.com'),('url','https://github.com/'),('url','https://docs.python.org/3/library/urllib.parse.html'),('sms','Your bank access expires in 30 minutes. Send your OTP and card PIN to this number to keep it active.')]:
    req=urllib.request.Request('http://localhost:8000/api/v1/analyze/'+kind,data=json.dumps({('url' if kind=='url' else 'text'):value}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as r:live=json.load(r)
    direct=service.analyze(kind,value)
    equal=all(live[k]==direct[k] for k in ['prediction','probability','model','model_evidence'])
    assert equal
    results.append({'input':value,'prediction':live['prediction'],'risk_score':live['risk_score'],'artifact_sha256':live['model']['artifact_sha256'],'equals_direct':equal})
for route,payload in [('/url',{'url':'not a url'}),('',{'input_type':'url','value':'javascript:alert(1)'})]:
    req=urllib.request.Request('http://localhost:8000/api/v1/analyze'+route,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:urllib.request.urlopen(req);raise AssertionError('Invalid input accepted')
    except urllib.error.HTTPError as e:assert e.code==422 and isinstance(json.load(e)['detail'],str)
before=json.loads((ROOT/'audit/history_before_repair_verification.json').read_text())
with SessionLocal() as s:
    rows=s.execute(select(Analysis.__table__).where(Analysis.id<=before['max_id']).order_by(Analysis.id)).mappings().all()
    digest=hashlib.sha256(json.dumps([dict(row) for row in rows],default=str,sort_keys=True).encode()).hexdigest()
assert digest==before['sha256'] and len(rows)==before['count']
report={'live_checks':results,'invalid_inputs_rejected':True,'existing_history_unchanged':True,'preserved_history_rows':len(rows)}
(ROOT/'audit/repair_live_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
