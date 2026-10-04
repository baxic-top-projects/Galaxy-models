import json
from pathlib import Path

nb = json.loads(Path(r"d:\GitHub\Galaxy-models\remote_jupyter.ipynb").read_text(encoding="utf-8"))
s18 = "".join(nb["cells"][18]["source"])
s22 = "".join(nb["cells"][22]["source"])
print("origin_joblib", "origin_model.joblib" in s22)
print("fit_origin_line", any("target_origin" in line and "fit_classifier" in line for line in s18.splitlines()))
print("origin_head_meta", "origin_head" in s22)
print("load_origin_artifact", 'context.artifacts["origin_model"]' in s18)
