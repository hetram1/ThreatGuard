import json
from pathlib import Path
from backend.app.services.threat_analysis_service import ThreatAnalysisService
import pandas as pd

root = Path.cwd()
service = ThreatAnalysisService()
probes = pd.read_csv(root / "audit/fresh_probes.csv").to_dict("records")
results = [
    {
        "kind": p["kind"],
        "input": p["value"],
        "result": service.analyze(p["kind"], p["value"]),
    }
    for p in probes
]
(root / "audit/cleanup_behavior_before.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)
print("Captured", len(results), "complete inference responses before refactoring")
