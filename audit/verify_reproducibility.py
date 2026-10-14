import json
from pathlib import Path
from ml.reliable.prepare import ROOT,sha
r=ROOT/'models/releases/reliable-v1';repeat=ROOT/'models/releases/reproducibility-v1'
result={name:sha(r/f'{name}.joblib')==sha(repeat/f'{name}.joblib') for name in ['url','sms']}
assert all(result.values()),result
(ROOT/'audit/reproducibility_verification.json').write_text(json.dumps({'independent_repeat_artifact_hashes_match':result,'method':'Repeated train/selection/calibration with unchanged prepared data, fixed candidate grid and seeds; no external metrics used.'},indent=2))
print(result)
# Preserve small provenance alongside the release for source control.
import shutil
shutil.copyfile(ROOT/'data/reliable/v2/manifest.json',r/'dataset_manifest.json')
shutil.copyfile(ROOT/'data/reliable/v2/legacy_artifacts.json',r/'legacy_artifacts.json')
