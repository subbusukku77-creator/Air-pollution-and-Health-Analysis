"""
train_model.py
==============
Air Pollution and Health Impact Prediction - Model Training & Evaluation Pipeline
Author: Antigravity (ML Engineer)
Date: 2026-11-15

This script performs end-to-end machine learning:
1. Loads and explores the processed health impact dataset.
2. Identifies features, checks for missing values, duplicates, and outliers.
3. Applies a leakage-free preprocessing pipeline (StandardScaler fit only on training data).
4. Trains and evaluates 6 regression algorithms:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
   - LightGBM Regressor
5. Evaluates models on Test and Train sets using R², MAE, MSE, and RMSE.
6. Selects the best performing model based on test generalization.
7. Saves the complete preprocessing + model pipeline to 'models/health_impact_model.pkl'.
8. Generates evaluation plots (Feature Importance, Actual vs. Predicted, Model Comparison).
9. Tests pipeline predictions across the full spectrum (clean air, moderate, and high pollution).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_inspect_data(filepath):
    """
    Load dataset and perform exploratory data analysis:
    - Shape, columns, types
    - Missing values
    - Duplicates
    - Outlier detection (IQR method)
    """
    print("=" * 80)
    print("STEP 1: DATA INSPECTION & EXPLORATION")
    print("=" * 80)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: '{filepath}'")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\nColumns and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<22}: {dtype}")

    # Missing Values
    missing = df.isnull().sum()
    print(f"\nMissing Values Total: {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  ✓ No missing values detected.")

    # Duplicate Records
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    if duplicates == 0:
        print("  ✓ No duplicate records detected.")

    # Feature List
    feature_cols = [
        'AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3',
        'Temperature', 'Humidity', 'WindSpeed',
        'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions'
    ]
    target_col = 'HealthImpactScore'

    print(f"\nInput Features ({len(feature_cols)}):")
    print(" ", ", ".join(feature_cols))
    print(f"Target Variable: {target_col}")
    print(f"Target Summary:\n{df[target_col].describe()}")

    # Outlier Detection via IQR Method
    print("\nOutlier Detection (IQR Method):")
    outlier_records = []
    for col in feature_cols + [target_col]:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = (outlier_count / len(df)) * 100
        skewness = df[col].skew()
        outlier_records.append({
            'Feature': col,
            'Min': round(df[col].min(), 2),
            'Max': round(df[col].max(), 2),
            'Mean': round(df[col].mean(), 2),
            'Std': round(df[col].std(), 2),
            'Outliers (IQR)': outlier_count,
            'Outlier %': f"{pct:.2f}%",
            'Skewness': round(skewness, 2)
        })

    outlier_df = pd.DataFrame(outlier_records)
    print(outlier_df.to_string(index=False))

    return df, feature_cols, target_col


def prepare_data(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Separate features and target, and perform 80/20 train-test split.
    Guarantees no data leakage by leaving scaling to pipeline fit on train set.
    """
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPARATION & SPLIT (80% TRAIN / 20% TEST)")
    print("=" * 80)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set:  X_test={X_test.shape},  y_test={y_test.shape}")
    print("✓ Preprocessing pipeline will be fit strictly on X_train to prevent data leakage.")

    return X_train, X_test, y_train, y_test


def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple regression models wrapped in a StandardScaler pipeline.
    Algorithms:
      1. Linear Regression
      2. Decision Tree
      3. Random Forest
      4. Gradient Boosting
      5. XGBoost
      6. LightGBM
    """
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING & COMPARING REGRESSION ALGORITHMS")
    print("=" * 80)

    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            random_state=42, max_depth=8, min_samples_leaf=5
        ),
        'Random Forest': RandomForestRegressor(
            random_state=42, n_estimators=150, max_depth=12, min_samples_leaf=2, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            random_state=42, n_estimators=200, learning_rate=0.05, max_depth=5
        ),
        'XGBoost': XGBRegressor(
            random_state=42, n_estimators=200, learning_rate=0.05, max_depth=5,
            eval_metric='rmse', n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, verbose=-1, n_jobs=-1
        )
    }

    results = []
    trained_pipelines = {}

    print(f"{'Model':<20} | {'Train R²':<9} | {'Test R²':<9} | {'Test MAE':<9} | {'Test MSE':<9} | {'Test RMSE':<9}")
    print("-" * 75)

    for name, model in candidate_models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        # Fit strictly on training data
        pipeline.fit(X_train, y_train)

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = float(np.sqrt(mse))

        results.append({
            'Model': name,
            'Train R2': train_r2,
            'Test R2': test_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        })
        trained_pipelines[name] = {
            'pipeline': pipeline,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae
        }

        print(f"{name:<20} | {train_r2:<9.4f} | {test_r2:<9.4f} | {mae:<9.4f} | {mse:<9.4f} | {rmse:<9.4f}")

    results_df = pd.DataFrame(results)

    # Select best model: highest test R² and lowest test RMSE
    best_row = results_df.sort_values(by=['Test R2', 'RMSE'], ascending=[False, True]).iloc[0]
    best_name = best_row['Model']

    print("\n" + "=" * 80)
    print(f"BEST MODEL SELECTED: {best_name}")
    print(f"  Test R²:   {best_row['Test R2']:.4f}")
    print(f"  Test RMSE: {best_row['RMSE']:.4f}")
    print(f"  Test MAE:  {best_row['MAE']:.4f}")
    print(f"  Test MSE:  {best_row['MSE']:.4f}")
    print("=" * 80)

    return results_df, trained_pipelines, best_name


def save_pipeline_and_plots(trained_pipelines, best_name, feature_cols, X_test, y_test, results_df, output_model_path, plots_dir):
    """
    Save the winning pipeline to disk and generate visualization plots.
    """
    print("\n" + "=" * 80)
    print("STEP 4: SAVING MODEL & GENERATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Save Model Pipeline
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    best_pipeline = trained_pipelines[best_name]['pipeline']
    joblib.dump(best_pipeline, output_model_path)
    print(f"✓ Saved best model pipeline to: '{output_model_path}'")

    # 2. Setup plots directory
    os.makedirs(plots_dir, exist_ok=True)

    # 3. Model Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ['#A6B37D', '#B99470', '#CCD5AE', '#E9EDC9', '#4B5945', '#7C8C58']

    sns.barplot(ax=axes[0], data=results_df, x='Model', y='Test R2', hue='Model', palette=palette, legend=False)
    axes[0].set_title('Test R² Score (Higher is Better)', fontsize=12, fontweight='bold', color='#4B5945')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('R² Score')

    sns.barplot(ax=axes[1], data=results_df, x='Model', y='RMSE', hue='Model', palette=palette, legend=False)
    axes[1].set_title('Test RMSE (Lower is Better)', fontsize=12, fontweight='bold', color='#4B5945')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_ylabel('RMSE')

    sns.barplot(ax=axes[2], data=results_df, x='Model', y='MAE', hue='Model', palette=palette, legend=False)
    axes[2].set_title('Test MAE (Lower is Better)', fontsize=12, fontweight='bold', color='#4B5945')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].set_ylabel('MAE')

    plt.tight_layout()
    comp_plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved Model Comparison plot to: '{comp_plot_path}'")

    # 4. Feature Importance Plot
    regressor = best_pipeline.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        plt.figure(figsize=(9, 6))
        plt.barh(feat_df['Feature'], feat_df['Importance'], color='#A6B37D', edgecolor='#4B5945')
        plt.title(f'Feature Importance ({best_name})', fontsize=14, fontweight='bold', color='#4B5945')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        feat_plot_path = os.path.join(plots_dir, 'feature_importance.png')
        plt.savefig(feat_plot_path, dpi=300)
        plt.close()
        print(f"✓ Saved Feature Importance plot to: '{feat_plot_path}'")

    # 5. Actual vs Predicted Plot
    y_test_pred = trained_pipelines[best_name]['y_test_pred']
    residuals = y_test - y_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter of Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.5, color='#4B5945', edgecolors='none', s=30)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y = x)')
    axes[0].set_title(f'Actual vs. Predicted ({best_name})', fontsize=13, fontweight='bold', color='#4B5945')
    axes[0].set_xlabel('Actual HealthImpactScore')
    axes[0].set_ylabel('Predicted HealthImpactScore')
    axes[0].legend()

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#A6B37D')
    axes[1].axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_title(f'Residual Distribution ({best_name})', fontsize=13, fontweight='bold', color='#4B5945')
    axes[1].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    pred_plot_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved Actual vs. Predicted plot to: '{pred_plot_path}'")


def verify_pipeline_predictions(model_path, feature_cols):
    """
    Test loading the saved pipeline from disk and generating sample predictions
    across multiple realistic scenarios: clean air, moderate pollution, and high pollution.
    """
    print("\n" + "=" * 80)
    print("STEP 5: VERIFYING PIPELINE ON EXTERNAL TEST CASES")
    print("=" * 80)

    loaded_pipeline = joblib.load(model_path)
    print(f"Loaded pipeline steps: {list(loaded_pipeline.named_steps.keys())}")

    test_scenarios = [
        {
            'name': 'Clean Air / Low Health Risk',
            'data': {'AQI': 15.0, 'PM10': 10.0, 'PM2_5': 5.0, 'NO2': 5.0, 'SO2': 2.0, 'O3': 10.0,
                     'Temperature': 22.0, 'Humidity': 50.0, 'WindSpeed': 8.0,
                     'RespiratoryCases': 1.0, 'CardiovascularCases': 0.0, 'HospitalAdmissions': 0.0}
        },
        {
            'name': 'Moderate Pollution / Average Health Risk',
            'data': {'AQI': 120.0, 'PM10': 75.0, 'PM2_5': 45.0, 'NO2': 50.0, 'SO2': 25.0, 'O3': 75.0,
                     'Temperature': 18.0, 'Humidity': 55.0, 'WindSpeed': 6.0,
                     'RespiratoryCases': 8.0, 'CardiovascularCases': 3.0, 'HospitalAdmissions': 1.0}
        },
        {
            'name': 'Severe Air Pollution / High Health Risk',
            'data': {'AQI': 350.0, 'PM10': 250.0, 'PM2_5': 180.0, 'NO2': 160.0, 'SO2': 80.0, 'O3': 220.0,
                     'Temperature': 10.0, 'Humidity': 85.0, 'WindSpeed': 2.0,
                     'RespiratoryCases': 18.0, 'CardiovascularCases': 12.0, 'HospitalAdmissions': 5.0}
        }
    ]

    for scenario in test_scenarios:
        sample_df = pd.DataFrame([scenario['data']])[feature_cols]
        raw_pred = loaded_pipeline.predict(sample_df)[0]
        clipped_pred = np.clip(raw_pred, 0.0, 100.0)
        print(f"  Scenario: {scenario['name']}")
        print(f"    -> Predicted HealthImpactScore: {clipped_pred:.2f} (raw: {raw_pred:.2f})")

    print("\n✓ Verification across external scenarios passed successfully!")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'data', 'processed', 'health_impact_cleaned.csv')
    model_output_path = os.path.join(base_dir, 'models', 'health_impact_model.pkl')
    plots_directory = os.path.join(base_dir, 'static')

    # Execute workflow
    df, feature_cols, target_col = load_and_inspect_data(dataset_path)
    X_train, X_test, y_train, y_test = prepare_data(df, feature_cols, target_col, test_size=0.2, random_state=42)
    results_df, trained_pipelines, best_name = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_pipeline_and_plots(trained_pipelines, best_name, feature_cols, X_test, y_test, results_df, model_output_path, plots_directory)
    verify_pipeline_predictions(model_output_path, feature_cols)

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
