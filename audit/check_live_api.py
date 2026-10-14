import sys,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services.threat_analysis_service import ThreatAnalysisService
service=ThreatAnalysisService();rows=[]
for kind,value in [('url','https://github.com'),('url','https://github.com/'),('url','not a url'),('sms','Your bank access expires in 30 minutes. Send your OTP and card PIN to this number to keep it active.')]:
    payload={('url' if kind=='url' else 'text'):value}
    req=urllib.request.Request('http://localhost:8000/api/v1/analyze/'+kind,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as response:
        live=json.load(response);status=response.status
    direct=service.analyze(kind,value)
    rows.append(dict(kind=kind,input=value,status=status,live=live,equals_direct=all(live[k]==direct[k] for k in ['prediction','risk_score','confidence','reasons'])))
(ROOT/'audit/live_api_checks.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
