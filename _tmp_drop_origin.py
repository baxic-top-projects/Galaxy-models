"""Drop origin_model from training/serving to fit 8GB stormmodel host."""
from __future__ import annotations

import json
from pathlib import Path

nb_path = Path(r"d:\GitHub\Galaxy-models\remote_jupyter.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))


def to_source(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").rstrip() + "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]


# --- cell 18: pyfunc + train ---
cell18 = '''from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def fit_classifier(features, target, *, seed=42, params=None):
    target = target.astype(str) if target.dtype == "object" else target
    if target.nunique() < 2:
        return DummyClassifier(strategy="prior").fit(features, target)
    model_params = {
        "n_estimators": 150,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": seed,
        "n_jobs": -1,
    }
    if params:
        model_params.update(params)
    model_params["random_state"] = seed
    model_params["n_jobs"] = -1
    model_params["n_estimators"] = min(int(model_params.get("n_estimators", 150)), 200)
    if model_params.get("max_depth") is not None:
        model_params["max_depth"] = min(int(model_params["max_depth"]), 14)
    return RandomForestClassifier(**model_params).fit(features, target)


def class_probability_maps(model, features):
    proba = model.predict_proba(features)
    classes = [str(c) for c in model.classes_]
    return [{c: float(p) for c, p in zip(classes, row)} for row in proba]


def positive_probability(model, features):
    maps = class_probability_maps(model, features)
    return np.asarray([m.get("1", 0.0) for m in maps], dtype=float)


class StormSpawnPyfunc(mlflow.pyfunc.PythonModel):
    """Serve spawn + type only. Origin (~7.5GB) is omitted for 8GB hosts."""

    def __init__(self, feature_columns=None, spawn_model=None, type_model=None, threshold=0.5):
        self.feature_columns = list(feature_columns or [])
        self.spawn_model = spawn_model
        self.type_model = type_model
        self.threshold = float(threshold)

    def load_context(self, context):
        meta = json.loads(Path(context.artifacts["meta"]).read_text(encoding="utf-8"))
        self.feature_columns = list(meta["feature_columns"])
        self.threshold = float(meta.get("threshold", 0.5))
        self.spawn_model = joblib.load(context.artifacts["spawn_model"])
        self.type_model = joblib.load(context.artifacts["type_model"])

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input).copy()
        missing = [c for c in self.feature_columns if c not in frame]
        if missing:
            raise ValueError(f"missing features: {missing}")
        features = frame[self.feature_columns].astype(float)
        spawn_p = positive_probability(self.spawn_model, features)
        type_maps = class_probability_maps(self.type_model, features)
        out = []
        for i, p in enumerate(spawn_p):
            type_map = type_maps[i]
            out.append(
                {
                    "spawn_probability": float(p),
                    "should_spawn": bool(p >= self.threshold),
                    "predicted_type": max(type_map, key=type_map.get) if type_map else None,
                    "predicted_origin": None,
                    "type_probabilities": json.dumps(type_map, sort_keys=True),
                    "origin_probabilities": "{}",
                }
            )
        return pd.DataFrame(out)


def train_compound_model(frame, *, spawn_params=None) -> StormSpawnPyfunc:
    features = frame[FEATURE_COLUMNS]
    positive = frame[frame["target_spawn"] == 1]
    if positive.empty:
        raise ValueError("no spawn events")
    return StormSpawnPyfunc(
        FEATURE_COLUMNS,
        fit_classifier(features, frame["target_spawn"], params=spawn_params),
        fit_classifier(positive[FEATURE_COLUMNS], positive["target_type"]),
    )


evaluation_model = train_compound_model(train_data, spawn_params=best_params)
validation_input = validation_data[FEATURE_COLUMNS].copy()
validation_predictions = evaluation_model.predict(None, validation_input)
print(validation_predictions.head())
free_ram("train/eval predict")
'''

# --- cell 20: eval without origin metric ---
src20 = "".join(nb["cells"][20]["source"])
old_origin_block = '''pos = validation_data["target_spawn"].to_numpy(dtype=bool)
if pos.any():
    model_metrics["model_type_accuracy"] = accuracy_score(
        validation_data.loc[pos, "target_type"].astype(str),
        validation_predictions.loc[pos, "predicted_type"].astype(str),
    )
    model_metrics["model_origin_top1_accuracy"] = accuracy_score(
        validation_data.loc[pos, "target_origin"].astype(str),
        validation_predictions.loc[pos, "predicted_origin"].astype(str),
    )
'''
new_origin_block = '''pos = validation_data["target_spawn"].to_numpy(dtype=bool)
if pos.any():
    model_metrics["model_type_accuracy"] = accuracy_score(
        validation_data.loc[pos, "target_type"].astype(str),
        validation_predictions.loc[pos, "predicted_type"].astype(str),
    )
# origin model intentionally omitted from serving (RAM); skip origin metrics
'''
if old_origin_block not in src20:
    raise SystemExit("origin metrics block not found in cell 20")
src20 = src20.replace(old_origin_block, new_origin_block)

# --- cell 22: register without origin artifact ---
src22 = "".join(nb["cells"][22]["source"])
replacements = [
    (
        '''final_model = train_compound_model(training_data, spawn_params=best_params)
known_origins = sorted(
    training_data.loc[training_data["target_spawn"] == 1, "target_origin"].dropna().astype(str).unique()
)
input_example = training_data.iloc[[-1]][FEATURE_COLUMNS].copy()
input_example["candidate_origins_json"] = json.dumps(known_origins[:100])
output_example = final_model.predict(None, input_example)
signature = infer_signature(input_example, output_example)
''',
        '''final_model = train_compound_model(training_data, spawn_params=best_params)
input_example = training_data.iloc[[-1]][FEATURE_COLUMNS].copy()
output_example = final_model.predict(None, input_example)
signature = infer_signature(input_example, output_example)
''',
    ),
    (
        '''    joblib.dump(final_model.spawn_model, local_dir / "spawn_model.joblib")
    joblib.dump(final_model.type_model, local_dir / "type_model.joblib")
    joblib.dump(final_model.origin_model, local_dir / "origin_model.joblib")
    (local_dir / "meta.json").write_text(
        json.dumps(
            {"feature_columns": FEATURE_COLUMNS, "threshold": final_model.threshold},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
''',
        '''    joblib.dump(final_model.spawn_model, local_dir / "spawn_model.joblib")
    joblib.dump(final_model.type_model, local_dir / "type_model.joblib")
    (local_dir / "meta.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "threshold": final_model.threshold,
                "heads": ["spawn", "type"],
                "origin_head": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
''',
    ),
    (
        '''        artifacts={
            "spawn_model": str(local_dir / "spawn_model.joblib"),
            "type_model": str(local_dir / "type_model.joblib"),
            "origin_model": str(local_dir / "origin_model.joblib"),
            "meta": str(local_dir / "meta.json"),
        },
''',
        '''        artifacts={
            "spawn_model": str(local_dir / "spawn_model.joblib"),
            "type_model": str(local_dir / "type_model.joblib"),
            "meta": str(local_dir / "meta.json"),
        },
''',
    ),
    (
        '    description="joblib artifacts via direct S3 upload",',
        '    description="spawn+type joblib artifacts (no origin head; fits 8GB serve)",',
    ),
]
for old, new in replacements:
    if old not in src22:
        raise SystemExit(f"block not found in cell 22:\n{old[:120]}")
    src22 = src22.replace(old, new)

nb["cells"][18]["source"] = to_source(cell18)
nb["cells"][18]["outputs"] = []
nb["cells"][18]["execution_count"] = None
nb["cells"][20]["source"] = to_source(src20)
nb["cells"][22]["source"] = to_source(src22)

# markdown note
md21 = "".join(nb["cells"][21].get("source", []))
if "origin" not in md21.lower() or "spawn" in md21.lower():
    nb["cells"][21]["source"] = to_source(
        """## 10. Register model

1) Save **spawn + type** joblib artifacts (origin head disabled — too large for 8GB serve)
2) Upload Vast → S3 multipart
3) Register in MLflow
4) Delete only old model versions (S3 + registry) and leftover local model folders
"""
    )

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("patched cells 18/20/21/22 — origin head removed")
