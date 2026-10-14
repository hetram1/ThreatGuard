import json,hashlib
from pathlib import Path
from sqlalchemy import select
from backend.app.database.database import SessionLocal
from backend.app.models.analysis import Analysis
p=Path('audit/history_before_repair_verification.json')
with SessionLocal() as s:
    rows=s.execute(select(Analysis.__table__).order_by(Analysis.id)).mappings().all()
    text=json.dumps([dict(row) for row in rows],default=str,sort_keys=True)
    info={'count':len(rows),'max_id':max((r['id'] for r in rows),default=0),'sha256':hashlib.sha256(text.encode()).hexdigest()}
    if p.exists():raise SystemExit('Preserving existing snapshot')
    p.write_text(json.dumps(info,indent=2));print(info)
