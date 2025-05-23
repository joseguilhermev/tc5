import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import optuna


def train_model(df):
    df = df.dropna(subset=["y"])
    X = df.drop("y", axis=1)
    y = df["y"]

    num_cols = ["remuneracao"]
    cat_cols = [col for col in X.columns if col not in num_cols]

    num_transformer = SimpleImputer(strategy="median")
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_pre, y_train)  # type: ignore

    def objective(trial):
        params = {
            "n_estimators": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test_pre)
        return f1_score(y_test, y_pred)  # type: ignore

    print("[⚙️] Otimizando hiperparâmetros com Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # type: ignore

    print(f"\nMelhor conjunto de hiperparâmetros: {study.best_params}")

    best_model = LGBMClassifier(
        **study.best_params, class_weight="balanced", random_state=42, n_jobs=-1
    )
    best_model.fit(X_resampled, y_resampled)

    y_pred = best_model.predict(X_test_pre)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))  # type: ignore

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": best_model, "preprocessor": preprocessor},
        "models/modelo_lgbm_optuna.pkl",
    )
    print(
        "Modelo LightGBM otimizado com Optuna salvo em: models/modelo_lgbm_optuna.pkl"
    )

    return best_model
