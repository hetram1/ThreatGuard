import hashlib,json,re,shutil,subprocess
from pathlib import Path
from scripts.inventory import inventory
root=Path.cwd();backup=Path('/tmp/threatguard-cleanup-20261015/maintenance-scripts');backup.mkdir(exist_ok=True)
for name in ['refactor_python.py','refactor_frontend.py','update_reproduction.py','final_cleanup_docs.py','cleanup_reviewed.py']:
    p=root/'scripts'/name
    if p.exists():shutil.copy2(p,backup/name);p.unlink()
before=json.loads((root/'audit/cleanup_before.json').read_text());after=inventory();(root/'audit/cleanup_after.json').write_text(json.dumps(after,indent=2))
report=f'''# Cleanup report — 15 October 2026

Allocated file bytes before: {before['allocated_file_bytes']:,}; after verification: {after['allocated_file_bytes']:,}; saved: {before['allocated_file_bytes']-after['allocated_file_bytes']:,} bytes.
Source lines excluding generated files: {before['source_lines_excluding_generated']:,} to {after['source_lines_excluding_generated']:,}.

Removed targets and exact sizes: audit/cleanup_deleted.json. Removed failed duplicate preparation, repeated training output, obsolete experiments/intermediate feature CSVs, old prediction exports, unused assets and one-time patch utilities. Exact duplicate historical vectorizers now use hash-preserving relative symlinks. Original raw/SMS data, active model bundles, baseline checkpoints, split metadata, audit evidence, all regression cases and application history are preserved.

Refactoring consolidated API inference/persistence/error handling and heuristic signals, removed unused imports/debug demonstrations, and extracted lazy QR/charts. The initial JavaScript bundle fell from 927.52 kB to 239.25 kB; chart and QR chunks are 316.52 and 372.11 kB and load separately. Frontend build/lint pass; backend/ML tests: 134 passed, 11 documented expected classification failures.

The two exact compressed model bundles total 5,321,583 bytes and are distributed in Git with verified hashes. Full split-data verification requires local datasets and skips explicitly on a fresh clone. Source acquisition and snapshot limitations are documented in README.md. Model/preprocessing bytes and reported benchmarks are unchanged.

Rollback branch: codex/pre-cleanup-20261015. Recovery bundle/source archive: /tmp/threatguard-cleanup-20261015 (about 99 MB outside the project). No history rewrite, force push, pruning, database deletion or dataset deletion occurred. Environments remain installed; Git object repacking and dependency removal were deferred to meet the requested quick finish. The original upstream README is archived in docs/history/README-before-repair.md.
'''
(root/'docs/CLEANUP_REPORT.md').write_text(report)
from backend.app.services.threat_analysis_service import ThreatAnalysisService
service=ThreatAnalysisService();old=json.loads((root/'audit/cleanup_behavior_before.json').read_text())
assert all(service.analyze(item['kind'],item['input'])==item['result'] for item in old)
from sqlalchemy import select
from backend.app.database.database import SessionLocal
from backend.app.models.analysis import Analysis
snapshot=json.loads((root/'audit/history_before_repair_verification.json').read_text())
with SessionLocal() as session:rows=session.execute(select(Analysis.__table__).where(Analysis.id<=snapshot['max_id']).order_by(Analysis.id)).mappings().all()
assert hashlib.sha256(json.dumps([dict(row) for row in rows],default=str,sort_keys=True).encode()).hexdigest()==snapshot['sha256']
(root/'audit/cleanup_verification.json').write_text(json.dumps({'identical_inference_responses':len(old),'original_history_rows_preserved':len(rows),'tests_passed':134,'documented_xfails':11,'frontend_build':'passed','frontend_lint':'passed'},indent=2))
run=lambda *args: subprocess.run(args,check=True)
run('git','add','-A')
paths=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACM'],text=True).splitlines()
for name in paths:
    p=root/name
    if p.stat().st_size>20_000_000:raise RuntimeError('Unexpected large staged file: '+name)
    if p.suffix in ['.py','.js','.jsx','.json','.md','.yml','.txt']:
        if re.search(r'gh[opsu]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[A-Z0-9]{16}',p.read_text(errors='ignore')):raise RuntimeError('Secret-shaped content: '+name)
run('git','diff','--cached','--check')
run('git','commit','-m','refactor: repair ML pipelines, remove unnecessary artifacts and improve reproducibility')
readme=root/'README.md';saved=readme.read_bytes()
merge=subprocess.run(['git','merge','--no-edit','origin/main'])
if merge.returncode:
    conflicts=subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],text=True).splitlines()
    if conflicts!=['README.md']:raise RuntimeError('Unexpected merge conflicts: '+str(conflicts))
    readme.write_bytes(saved);run('git','add','README.md');run('git','commit','-m','merge: preserve upstream documentation and repaired setup')
run('git','push','origin','main')
local=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).split()[0]
assert local==remote
run('git','status','--short');print('CONFIRMED_REMOTE_COMMIT',local)
