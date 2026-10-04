import json
from pathlib import Path

nb_path = Path(r"D:\GitHub\Galaxy-models\remote_jupyter.ipynb")
nb = json.load(nb_path.open(encoding="utf-8"))


def to_source(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").rstrip() + "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]


optuna_cell = '''import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
X_train = train_data[FEATURE_COLUMNS]
y_train = train_data["target_spawn"].astype(int)
X_valid = validation_data[FEATURE_COLUMNS]
y_valid = validation_data["target_spawn"].astype(int)


def objective(trial: optuna.Trial) -> float:
    params = {
        # Keep forests small enough for Cloudflare's ~100 MB upload limit.
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced_subsample", "balanced", None]
        ),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)
    # Keep class order stable for binary labels.
    class_index = {str(label): index for index, label in enumerate(model.classes_)}
    ordered = np.column_stack(
        [
            probabilities[:, class_index.get("0", 0)],
            probabilities[:, class_index.get("1", min(1, probabilities.shape[1] - 1))],
        ]
    )
    return float(log_loss(y_valid, ordered, labels=[0, 1]))


study = optuna.create_study(direction="minimize", study_name="storm-spawn-rf")
study.optimize(objective, n_trials=int(os.getenv("OPTUNA_TRIALS", "25")), show_progress_bar=True)
best_params = dict(study.best_params)
best_params.update({"random_state": 42, "n_jobs": -1})

print("Best validation log-loss:", round(study.best_value, 6))
print("Best params:")
print(json.dumps(best_params, indent=2, ensure_ascii=False))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()
'''

model_cell = '''from pathlib import Path
import tempfile

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def fit_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int = 42,
    params: dict | None = None,
):
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
    # Hard cap so a stale Optuna best_params cannot rebuild a 100MB+ pickle.
    if "n_estimators" in model_params:
        model_params["n_estimators"] = min(int(model_params["n_estimators"]), 200)
    if model_params.get("max_depth") is not None:
        model_params["max_depth"] = min(int(model_params["max_depth"]), 16)
    return RandomForestClassifier(**model_params).fit(features, target)


def class_probability_maps(model, features: pd.DataFrame) -> list[dict[str, float]]:
    probabilities = model.predict_proba(features)
    classes = [str(value) for value in model.classes_]
    return [
        {label: float(probability) for label, probability in zip(classes, row)}
        for row in probabilities
    ]


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    probability_maps = class_probability_maps(model, features)
    return np.asarray([mapping.get("1", 0.0) for mapping in probability_maps], dtype=float)


class StormSpawnPyfunc(mlflow.pyfunc.PythonModel):
    """Compound model; heavy sklearn objects live in compressed artifacts for MLflow upload."""

    def __init__(
        self,
        feature_columns=None,
        spawn_model=None,
        type_model=None,
        origin_model=None,
        threshold=0.5,
    ):
        self.feature_columns = list(feature_columns or [])
        self.spawn_model = spawn_model
        self.type_model = type_model
        self.origin_model = origin_model
        self.threshold = float(threshold)

    def load_context(self, context):
        meta_path = Path(context.artifacts["meta"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_columns = list(meta["feature_columns"])
        self.threshold = float(meta.get("threshold", 0.5))
        self.spawn_model = joblib.load(context.artifacts["spawn_model"])
        self.type_model = joblib.load(context.artifacts["type_model"])
        self.origin_model = joblib.load(context.artifacts["origin_model"])

    @staticmethod
    def _candidate_set(value) -> set[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return set()
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else []
        return {str(item) for item in value}

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input).copy()
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        features = frame[self.feature_columns].astype(float)
        spawn_probabilities = positive_probability(self.spawn_model, features)
        type_maps = class_probability_maps(self.type_model, features)
        origin_maps = class_probability_maps(self.origin_model, features)

        output = []
        for index, spawn_probability in enumerate(spawn_probabilities):
            type_map = type_maps[index]
            origin_map = origin_maps[index]
            candidates = self._candidate_set(frame.iloc[index].get("candidate_origins_json"))
            if candidates:
                filtered = {key: value for key, value in origin_map.items() if key in candidates}
                total = sum(filtered.values()) or 1.0
                origin_map = {key: value / total for key, value in filtered.items()}
            ordered_origins = dict(
                sorted(origin_map.items(), key=lambda item: item[1], reverse=True)[:20]
            )
            predicted_type = max(type_map, key=type_map.get) if type_map else None
            predicted_origin = max(ordered_origins, key=ordered_origins.get) if ordered_origins else None
            output.append(
                {
                    "spawn_probability": float(spawn_probability),
                    "should_spawn": bool(spawn_probability >= self.threshold),
                    "predicted_type": predicted_type,
                    "predicted_origin": predicted_origin,
                    "type_probabilities": json.dumps(type_map, sort_keys=True),
                    "origin_probabilities": json.dumps(ordered_origins, sort_keys=True),
                }
            )
        return pd.DataFrame(output)


def train_compound_model(
    frame: pd.DataFrame,
    *,
    spawn_params: dict | None = None,
) -> StormSpawnPyfunc:
    features = frame[FEATURE_COLUMNS]
    positive = frame[frame["target_spawn"] == 1]
    if positive.empty:
        raise ValueError("No spawn events available")
    return StormSpawnPyfunc(
        FEATURE_COLUMNS,
        fit_classifier(features, frame["target_spawn"], params=spawn_params),
        fit_classifier(positive[FEATURE_COLUMNS], positive["target_type"]),
        fit_classifier(positive[FEATURE_COLUMNS], positive["target_origin"]),
    )


def export_compound_artifacts(model: StormSpawnPyfunc, directory: Path) -> dict[str, str]:
    """Write compressed joblib artifacts so Cloudflare can accept the MLflow upload."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "spawn_model": directory / "spawn_model.joblib",
        "type_model": directory / "type_model.joblib",
        "origin_model": directory / "origin_model.joblib",
        "meta": directory / "meta.json",
    }
    joblib.dump(model.spawn_model, paths["spawn_model"], compress=3)
    joblib.dump(model.type_model, paths["type_model"], compress=3)
    joblib.dump(model.origin_model, paths["origin_model"], compress=3)
    paths["meta"].write_text(
        json.dumps(
            {
                "feature_columns": list(model.feature_columns),
                "threshold": float(model.threshold),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sizes = {name: path.stat().st_size / (1024 * 1024) for name, path in paths.items()}
    print("Artifact sizes MB:", {k: round(v, 2) for k, v in sizes.items()})
    oversized = [name for name, size in sizes.items() if size >= 95]
    if oversized:
        raise RuntimeError(
            "Artifacts still too large for Cloudflare (~100 MB): "
            + ", ".join(oversized)
            + ". Reduce n_estimators / max_depth."
        )
    return {name: str(path) for name, path in paths.items()}


evaluation_model = train_compound_model(train_data, spawn_params=best_params)
validation_input = validation_data[FEATURE_COLUMNS].copy()
validation_input["candidate_origins_json"] = "[]"
validation_predictions = evaluation_model.predict(None, validation_input)
'''

mlflow_cell = '''import sklearn
from mlflow.models import infer_signature

final_model = train_compound_model(training_data, spawn_params=best_params)
known_origins = sorted(
    training_data.loc[training_data["target_spawn"] == 1, "target_origin"].dropna().astype(str).unique()
)
input_example = training_data.iloc[[-1]][FEATURE_COLUMNS].copy()
input_example["candidate_origins_json"] = json.dumps(known_origins[:100])
output_example = final_model.predict(None, input_example)
signature = infer_signature(input_example, output_example)

artifact_tmpdir = Path(tempfile.mkdtemp(prefix="storm-model-artifacts-"))
model_artifacts = export_compound_artifacts(final_model, artifact_tmpdir)
# Empty shell keeps python_model.pkl tiny; forests load from compressed artifacts.
logged_model = StormSpawnPyfunc()

mlflow.set_experiment("galaxy-storm-spawn")
with mlflow.start_run(run_name=f"storm-spawn-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}") as run:
    mlflow.set_tags(
        {
            "model_role": "storm-spawn-policy",
            "data_source": "airflow-s3-galaxy.storms",
            "airflow_dag": DAG_ID,
            "serving_alias": MODEL_ALIAS,
        }
    )
    mlflow.log_params(
        {
            "model_family": "compound-random-forest",
            "feature_count": len(FEATURE_COLUMNS),
            "training_rows": len(training_data),
            "spawn_events": int(training_data["target_spawn"].sum()),
            "origin_classes": len(known_origins),
            "time_split": 0.8,
            "random_state": 42,
            "optuna_trials": len(study.trials),
            "optuna_best_value": study.best_value,
            **{f"spawn_{key}": value for key, value in best_params.items() if value is not None},
        }
    )
    mlflow.log_metrics(
        {
            key: float(value)
            for key, value in all_metrics.items()
            if value is not None and np.isfinite(value)
        }
    )
    mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="evaluation")
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=logged_model,
        artifacts=model_artifacts,
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow==2.18.0",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"joblib=={joblib.__version__}",
        ],
    )
    run_id = run.info.run_id

client = MlflowClient()
matching_versions = [
    version
    for version in client.search_model_versions(f"name = '{MODEL_NAME}'")
    if version.run_id == run_id
]
if not matching_versions:
    raise RuntimeError(f"MLflow did not register a version for run {run_id}")
model_version = max(matching_versions, key=lambda item: int(item.version))
client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, model_version.version)
client.set_model_version_tag(MODEL_NAME, model_version.version, "validation", "chronological")

production_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
loaded_model = mlflow.pyfunc.load_model(production_uri)
local_smoke = loaded_model.predict(input_example)
print("MLflow run:", run_id)
print("Registered model:", MODEL_NAME)
print("Version:", model_version.version)
print("Alias:", MODEL_ALIAS)
print("Local registry smoke test:")
display(local_smoke)
'''

patched = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") != "code":
        continue
    if "study = optuna.create_study" in src and "storm-spawn-rf" in src:
        nb["cells"][i]["source"] = to_source(optuna_cell)
        patched.append(("optuna", i))
    elif "class StormSpawnPyfunc" in src:
        nb["cells"][i]["source"] = to_source(model_cell)
        patched.append(("model", i))
    elif "mlflow.pyfunc.log_model" in src and "registered_model_name=MODEL_NAME" in src:
        nb["cells"][i]["source"] = to_source(mlflow_cell)
        patched.append(("mlflow", i))

print(patched)
if len(patched) != 3:
    raise SystemExit(f"expected 3 patches, got {patched}")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("done")
