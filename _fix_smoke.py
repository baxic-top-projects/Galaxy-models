import json
from pathlib import Path

nb_path = Path(r"D:\GitHub\Galaxy-models\remote_jupyter.ipynb")
nb = json.load(nb_path.open(encoding="utf-8"))


def to_source(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").rstrip() + "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]


smoke = '''STORMMODEL_USERNAME = require_env("STORMMODEL_USERNAME", default="model-api")
STORMMODEL_PASSWORD = require_env("STORMMODEL_PASSWORD")
if STORMMODEL_PASSWORD == "change-me":
    raise ValueError(
        "STORMMODEL_PASSWORD is still change-me. "
        "Put the nginx htpasswd password for model-api into .env and re-run credentials."
    )
endpoint_auth = (STORMMODEL_USERNAME, STORMMODEL_PASSWORD)
invocation_payload = {"dataframe_records": input_example.to_dict(orient="records")}

# Large s3:// model download on the serving host can take many minutes.
wait_seconds = int(os.getenv("STORMMODEL_READY_TIMEOUT", "900"))
deadline = time.monotonic() + wait_seconds
last_error = None
endpoint_prediction = None
print(f"Waiting up to {wait_seconds}s for {MODEL_ENDPOINT} ...")
while time.monotonic() < deadline:
    try:
        ping = requests.get(f"{MODEL_ENDPOINT}/ping", auth=endpoint_auth, timeout=20)
    except requests.RequestException as exc:
        last_error = f"Ping error: {exc}"
        time.sleep(5)
        continue
    if ping.status_code in {401, 403}:
        raise PermissionError(
            "Storm model endpoint rejected Basic Auth. "
            "Check STORMMODEL_USERNAME / STORMMODEL_PASSWORD against /etc/nginx/.htpasswd-stormmodel"
        )
    if ping.ok:
        try:
            invocation = requests.post(
                f"{MODEL_ENDPOINT}/invocations",
                auth=endpoint_auth,
                json=invocation_payload,
                timeout=120,
            )
        except requests.RequestException as exc:
            last_error = f"Invocation error: {exc}"
            time.sleep(5)
            continue
        if invocation.ok:
            endpoint_prediction = invocation.json()
            break
        last_error = f"HTTP {invocation.status_code}: {invocation.text[:300]}"
    else:
        last_error = f"Ping HTTP {ping.status_code}: {ping.text[:300]}"
    time.sleep(5)

if endpoint_prediction is None:
    raise TimeoutError(
        f"Serving did not become ready for {MODEL_NAME}@{MODEL_ALIAS} within {wait_seconds}s. "
        f"Last error: {last_error}"
    )

deployment_summary = {
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "model_version": str(model_version.version),
    "alias": MODEL_ALIAS,
    "endpoint": MODEL_ENDPOINT,
    "ping": "ok",
    "invocation": endpoint_prediction,
}
with mlflow.start_run(run_id=run_id):
    mlflow.log_dict(deployment_summary, "deployment/smoke-test.json")

print(json.dumps(deployment_summary, ensure_ascii=False, indent=2))
'''

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "STORMMODEL_USERNAME = credential" in src and "/invocations" in src:
        nb["cells"][i]["source"] = to_source(smoke)
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print("patched smoke", i)
        break
else:
    raise SystemExit("smoke cell not found")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("done")
