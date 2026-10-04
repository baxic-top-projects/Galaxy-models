import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(Path(".env"), override=True)
base = os.environ["AIRFLOW_URL"].rstrip("/")
auth = (os.environ["AIRFLOW_USERNAME"], os.environ["AIRFLOW_PASSWORD"])
s = requests.Session()
s.auth = auth
s.headers["Accept"] = "application/json"

# recent runs timing
runs = s.get(
    f"{base}/api/v1/dags/prepare_galaxy_logs_for_jupyter/dagRuns",
    params={"limit": 5, "order_by": "-start_date"},
    timeout=20,
).json()["dag_runs"]
for run in runs:
    print(
        run.get("state"),
        run.get("dag_run_id"),
        "start=",
        run.get("start_date"),
        "end=",
        run.get("end_date"),
    )

# trigger one small page and measure
dag_run_id = f"probe__{int(time.time())}"
t0 = time.perf_counter()
resp = s.post(
    f"{base}/api/v1/dags/prepare_galaxy_logs_for_jupyter/dagRuns",
    json={"dag_run_id": dag_run_id, "conf": {"max_files": 5, "last_key": ""}},
    timeout=20,
)
print("trigger", resp.status_code, round(time.perf_counter() - t0, 2), "s")
resp.raise_for_status()

encoded = quote(dag_run_id, safe="")
while time.perf_counter() - t0 < 180:
    st = s.get(
        f"{base}/api/v1/dags/prepare_galaxy_logs_for_jupyter/dagRuns/{encoded}",
        timeout=20,
    ).json()
    state = st.get("state")
    print("state", state, round(time.perf_counter() - t0, 2), "s")
    if state in {"success", "failed", "skipped"}:
        break
    time.sleep(2)

x = s.get(
    f"{base}/api/v1/dags/prepare_galaxy_logs_for_jupyter/dagRuns/{encoded}/taskInstances/create_download_manifest/xcomEntries/return_value",
    params={"deserialize": "false"},
    timeout=30,
)
print("xcom", x.status_code)
if x.ok:
    value = x.json().get("value")
    if isinstance(value, str):
        value = json.loads(value)
    files = value.get("files", []) if isinstance(value, dict) else []
    print("files", len(files))
    if files:
        print("sample_keys", sorted(files[0].keys()))
print("total", round(time.perf_counter() - t0, 2), "s")
