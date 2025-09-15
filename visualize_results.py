import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import shap
import os
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

sns.set(style="whitegrid")

def load_data():
    # if you have a 'date' column, it'll be used for the time-series plot
    df = pd.read_csv('X_prepared.csv')
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        X = df.drop(columns=['date']).values
        feature_names = df.drop(columns=['date']).columns.tolist()
    else:
        dates = None
        X = df.values
        feature_names = df.columns.tolist()
    y = pd.read_csv('y_prepared.csv').values.flatten()
    return dates, X, y, feature_names

def load_models():
    rnn  = load_model('best_rnn_model.h5', compile=False)
    xgb  = joblib.load('best_xgb_model.pkl')
    lgbm = joblib.load('best_lgbm_model.pkl')
    # for stacking meta‑learner weights (optional)
    meta = joblib.load('stacking_meta.pkl') if os.path.exists('stacking_meta.pkl') else None
    return rnn, xgb, lgbm, meta

def get_preds(rnn, xgb, lgbm, X):
    scaler = MinMaxScaler().fit(X)
    X_s = scaler.transform(X)
    X_rnn = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
    p_rnn  = rnn.predict(X_rnn).flatten()
    p_xgb  = xgb.predict(X_s)
    p_lgbm = lgbm.predict(X_s)
    return p_rnn, p_xgb, p_lgbm

def plot_time_series(dates, y_true, y_pred, title):
    plt.figure(figsize=(12,4))
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
    else:
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.show()

def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout(); plt.show()

def plot_residuals(dates, y_true, y_pred):
    res = y_true - y_pred
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sns.histplot(res, bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Residual Distribution')
    if dates is not None:
        ax[1].plot(dates, res, '.', alpha=0.5)
        ax[1].set_title('Residuals Over Time')
    else:
        ax[1].scatter(range(len(res)), res, alpha=0.5)
        ax[1].set_title('Residuals vs Index')
    plt.tight_layout(); plt.show()

def plot_feature_importance(xgb, lgbm, feature_names):
    ix = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
    il = pd.Series(lgbm.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    sns.barplot(x=ix.values, y=ix.index, ax=axes[0], palette='Blues_d')
    axes[0].set_title('XGB Feature Importance')
    sns.barplot(x=il.values, y=il.index, ax=axes[1], palette='Greens_d')
    axes[1].set_title('LGBM Feature Importance')
    plt.tight_layout(); plt.show()

def plot_shap_summary(xgb, X, feature_names):
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=True)

def plot_meta_weights(meta):
    coefs = pd.Series(meta.coef_, index=['RNN', 'XGB', 'LGBM'])
    plt.figure(figsize=(6,4))
    sns.barplot(x=coefs.values, y=coefs.index, palette='Purples_d')
    plt.title('Stacking Meta-Learner Weights')
    plt.tight_layout(); plt.show()

def plot_cv_score_boxplot(xgb, lgbm, X, y):
    scaler = MinMaxScaler().fit(X)
    X_s = scaler.transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = {
        'XGB': np.sqrt(-cross_val_score(xgb, X_s, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)),
        'LGBM': np.sqrt(-cross_val_score(lgbm, X_s, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1))
    }
    df = pd.DataFrame(scores)
    df_melt = df.melt(var_name='Model', value_name='RMSE')
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Model', y='RMSE', data=df_melt, palette='pastel')
    plt.title('CV RMSE by Model')
    plt.tight_layout(); plt.show()

def plot_equity_curve(dates, y_true, y_pred):
    signals = np.sign(y_pred)
    strat_returns = signals * y_true
    cum_returns = np.cumsum(strat_returns)
    plt.figure(figsize=(12,4))
    if dates is not None:
        plt.plot(dates, cum_returns, label='Strategy')
        plt.plot(dates, np.cumsum(y_true), label='Buy & Hold')
    else:
        plt.plot(cum_returns, label='Strategy')
        plt.plot(np.cumsum(y_true), label='Buy & Hold')
    plt.title('Equity Curve')
    plt.legend()
    plt.tight_layout(); plt.show()

def main():
    dates, X, y, feats = load_data()
    rnn, xgb, lgbm, meta = load_models()
    pr, px, pl = get_preds(rnn, xgb, lgbm, X)
    # pick your “final” prediction (here we show stacking if available)
    if meta is not None:
        stack1 = np.vstack([pr, px, pl]).T
        final = meta.predict(stack1)
        print(f"Stacking RMSE: {np.sqrt(mean_squared_error(y, final)):.4f}")
    else:
        final = (pr + px + pl) / 3
        print(f"Ensemble RMSE: {np.sqrt(mean_squared_error(y, final)):.4f}")

    # 1) time-series
    plot_time_series(dates, y, final, 'Actual vs. Predicted')

    # 2) scatter
    plot_scatter(y, final, 'Predicted vs. Actual')

    # 3) residuals
    plot_residuals(dates, y, final)

    # 4) feature importances
    plot_feature_importance(xgb, lgbm, feats)

    # 5) shap summary (tree-based)
    plot_shap_summary(xgb, X, feats)

    # 6) Meta-learner weights
    if meta is not None:
        plot_meta_weights(meta)

    # 7) CV RMSE boxplot
    plot_cv_score_boxplot(xgb, lgbm, X, y)

    # 8) Equity curve
    plot_equity_curve(dates, y, final)

if __name__ == "__main__":
    main()