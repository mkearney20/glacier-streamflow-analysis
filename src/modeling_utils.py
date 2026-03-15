from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

def kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def pbias(y_true, y_pred):
    return 100 * (np.sum(y_pred - y_true) / np.sum(y_true))

def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "PBIAS": pbias(y_true, y_pred)
    }

def prepare_data(df, features, target='log_q_cms', split_ratio=0.8):
    df = df.sort_index()
    X = df[features]
    y = df[target]
    split_idx = int(len(X) * split_ratio)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test_raw.index)
    return X_train, X_test, y_train, y_test, scaler

def tune_rf_model(X_train, y_train, n_iter=50, n_splits=5, random_state=42):
    param_dist = {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [5, 10, 15],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 5],
        "max_features": ["sqrt", "log2", 0.5]
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        cv=tscv, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1, random_state=random_state
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def train_predict_model(X_train, X_test, y_train, y_test, model_type='MLR', rf_tune=True):
    if model_type == 'MLR':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = evaluate(np.expm1(y_test), y_pred)
        return model, y_pred, metrics
    elif model_type == 'RF':
        if rf_tune:
            model = tune_rf_model(X_train, y_train)
        else:
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = evaluate(np.expm1(y_test), y_pred)
        return model, y_pred, metrics
    
def get_full_report(model, X_train, y_train, X_test, y_test, name):
    y_train_pred = np.expm1(model.predict(X_train))
    y_train_true = np.expm1(y_train)
    train_metrics = evaluate(y_train_true, y_train_pred)

    y_test_pred = np.expm1(model.predict(X_test))
    y_test_true = np.expm1(y_test)
    test_metrics = evaluate(y_test_true, y_test_pred)

    report = pd.DataFrame({
        "Metric": list(train_metrics.keys()),
        "Train": list(train_metrics.values()),
        "Test": list(test_metrics.values())
    })
    report["Difference"] = report["Train"] - report["Test"]
    print(f"\n{name} Performance Report")
    display(report)
    return report