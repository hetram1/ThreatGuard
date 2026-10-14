import requests,json
from pathlib import Path
out=Path('data/external');out.mkdir(exist_ok=True)
url='https://zenodo.org/api/records/8041387'
try:
    response=requests.get(url,timeout=25);response.raise_for_status();obj=response.json()
    info={'url':url,'title':obj['metadata'].get('title'),'license':obj['metadata'].get('license'),'description':obj['metadata'].get('description'),'files':[{k:f.get(k) for k in ['key','size','checksum','links']} for f in obj.get('files',[])]}
    print(json.dumps(info,indent=2));(out/'zenodo_metadata.json').write_text(json.dumps(info,indent=2))
except Exception as e: print(type(e).__name__,str(e))
