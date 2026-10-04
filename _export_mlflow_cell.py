import json
from pathlib import Path

nb = json.load(Path(r"D:\GitHub\Galaxy-models\remote_jupyter.ipynb").open(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    s = "".join(c.get("source", []))
    if "upload_directory_to_s3" in s and "direct-s3-multipart" in s:
        Path(r"D:\GitHub\Galaxy-models\_mlflow_cell.py").write_text(s, encoding="utf-8")
        print("exported", i, "chars", len(s))
        break
else:
    raise SystemExit("not found")
