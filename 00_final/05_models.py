import re
import os
import joblib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def data_split(data, method):
    data = pd.get_dummies(data=data, columns=['구분'])

    lst = [col for col in data.columns if re.search('max|min|median|mean|rainfall|tavg|humid', col)]
    sor = [col for col in data.columns if '구분' in col]

    X_train, X_test, y_train, y_test = [], [], [], []

    if method == 'random':
        #------ 학습 데이터와 테스트 데이터로 분할

        X = data[lst + sor]
        y = data['값']

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # X.fillna(X.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        #--------------------------------
        
    elif method in ['해남군', '김제시', '부안군']:
        #------선택한 지역의 데이터를 테스트데이터로
        test_df = data[data['시군구'] == method]
        train_df = data[data['시군구'] != method]
    
        X_train = train_df[lst + sor]
        y_train = train_df['값']
    
        X_test = test_df[lst + sor]
        y_test = test_df['값']
        #--------------------------------
        
    elif method == 'year':
        # ------2021년을 테스트데이터로
        test_df = data[data['year'] == 2021]
        train_df = data[data['year'] != 2021]

        X_train = train_df[lst + sor]
        y_train = train_df['값']
    
        X_test = test_df[lst + sor]
        y_test = test_df['값']
        # --------------------------------

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X_train, X_test, y_train, y_test

def train_fit_model(data, method, save_dir):
    rf_model_path = os.path.join(save_dir, f'rf_model_{method}.pkl')
    xgb_model_path = os.path.join(save_dir, f'xgb_model_{method}.pkl')

    X_train, X_test, y_train, y_test = data_split(data, method)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not os.path.exists(rf_model_path):

        # ------XGBoostRegressor 모델 학습
        xgb_regressor = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=10,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        xgb_regressor.fit(X_train_scaled, y_train)
        joblib.dump(xgb_regressor, xgb_model_path)

        # --------------------------------

        # ------RandomForestRegressor 모델 학습
        rf_regressor = RandomForestRegressor(n_estimators=1200, random_state=42)
        rf_regressor.fit(X_train_scaled, y_train)
        joblib.dump(rf_regressor, rf_model_path)
        # --------------------------------


    else:
        rf_regressor = joblib.load(rf_model_path)
        xgb_regressor = joblib.load(xgb_model_path)

    # ------예측
    xgb_y_pred = xgb_regressor.predict(X_test_scaled)
    xgb_mse = mean_squared_error(y_test, xgb_y_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_y_pred)

    rf_y_pred = rf_regressor.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_y_pred)

    # -------결과 그림
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 첫 번째 서브플롯 - RandomForest
    rf_y_train_pred = rf_regressor.predict(X_train_scaled)
    rf_y_test_pred = rf_regressor.predict(X_test_scaled)
    sns.scatterplot(x=y_train, y=rf_y_train_pred, ax=ax[0], label='Training Data', color='blue')
    sns.scatterplot(x=y_test, y=rf_y_test_pred, ax=ax[0], label='Test Data', color='orange')
    ax[0].plot([0, max(max(y_test), max(rf_y_test_pred))], [0, max(max(y_test), max(rf_y_test_pred))], 'k--')
    ax[0].set_xlim([-1, max(max(rf_y_test_pred), max(y_test)) * 1.05])
    ax[0].set_ylim([-1, max(max(rf_y_test_pred), max(y_test)) * 1.05])
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")
    ax[0].legend(loc='upper left')
    ax[0].set_title(f"RandomForest\n RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")

    # 두 번째 서브플롯 - XGBoost
    xgb_y_train_pred = xgb_regressor.predict(X_train_scaled)
    xgb_y_test_pred = xgb_regressor.predict(X_test_scaled)
    sns.scatterplot(x=y_train, y=xgb_y_train_pred, ax=ax[1], label='Training Data', color='blue')
    sns.scatterplot(x=y_test, y=xgb_y_test_pred, ax=ax[1], label='Test Data', color='orange')
    ax[1].plot([0, max(max(y_test), max(xgb_y_test_pred))], [0, max(max(y_test), max(xgb_y_test_pred))], 'k--')
    ax[1].set_xlim([-1, max(max(xgb_y_test_pred), max(y_test)) * 1.05])
    ax[1].set_ylim([-1, max(max(xgb_y_test_pred), max(y_test)) * 1.05])
    ax[1].set_xlabel("Actual")
    ax[1].set_ylabel("Predicted")
    ax[1].legend(loc='upper left')
    ax[1].set_title(f"XGBoost\n RMSE: {xgb_rmse:.4f} | R²: {xgb_r2:.4f}")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'model_result_{method}.png')
    plt.savefig(save_path)



def main():
    output_dir = './output'
    data_dir = os.path.join(output_dir, 'model_process')
    save_dir = os.path.join(output_dir,'model_results')
    os.makedirs(save_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, '전체_데이터.csv'))
    data = data[(data['항목'] == '생산량') & (data['구분'] != '합계')]
    data = data[data['구분'] != '밭벼']
    #
    methods = ['year', 'random', '해남군']

    for method in methods:
        train_fit_model(data, method, save_dir)
        print('Model fitting and evaluation completed for', method)


if __name__ == '__main__':
    main()