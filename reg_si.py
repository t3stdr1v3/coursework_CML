import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# загрузка и очистка данных
df = pd.read_excel('data/Данные_для_курсовои_Классическое_МО.xlsx')
df = df.drop('Unnamed: 0', axis=1)
df = df.dropna()
df = df.drop_duplicates()

zero_var_cols = ['NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_N_O', 
                 'fr_SH', 'fr_azide', 'fr_barbitur', 'fr_benzodiazepine', 
                 'fr_diazo', 'fr_dihydropyridine', 'fr_isocyan', 'fr_isothiocyan', 
                 'fr_lactam', 'fr_nitroso', 'fr_phos_acid', 'fr_phos_ester', 
                 'fr_prisulfonamd', 'fr_thiocyan']
df = df.drop(zero_var_cols, axis=1)

# таргет (логарифм SI)
y = np.log1p(df['SI'])

# признаки (исключаем IC50 и CC50 обязательно, иначе утечка данных)
exclude_cols = ['IC50, mM', 'CC50, mM', 'SI']
X = df.drop(exclude_cols, axis=1)

# сплит
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# словарь результатов
results = {}

# линейная регрессия
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['LinearRegression'] = {
    'R2': r2_score(y_test, y_pred_lr),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
}

# ridge
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()
ridge_gs = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2')
ridge_gs.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_gs.predict(X_test_scaled)
results['Ridge'] = {
    'R2': r2_score(y_test, y_pred_ridge),
    'MAE': mean_absolute_error(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'best_params': ridge_gs.best_params_
}

# lasso
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
lasso = Lasso(max_iter=5000)
lasso_gs = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
lasso_gs.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_gs.predict(X_test_scaled)
results['Lasso'] = {
    'R2': r2_score(y_test, y_pred_lasso),
    'MAE': mean_absolute_error(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'best_params': lasso_gs.best_params_
}

# random forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
rf = RandomForestRegressor(random_state=42)
rf_gs = GridSearchCV(rf, rf_params, cv=5, scoring='r2')
rf_gs.fit(X_train_scaled, y_train)
y_pred_rf = rf_gs.predict(X_test_scaled)
results['RandomForest'] = {
    'R2': r2_score(y_test, y_pred_rf),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'best_params': rf_gs.best_params_
}

# gradient boosting
gb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
gb = GradientBoostingRegressor(random_state=42)
gb_gs = GridSearchCV(gb, gb_params, cv=5, scoring='r2')
gb_gs.fit(X_train_scaled, y_train)
y_pred_gb = gb_gs.predict(X_test_scaled)
results['GradientBoosting'] = {
    'R2': r2_score(y_test, y_pred_gb),
    'MAE': mean_absolute_error(y_test, y_pred_gb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'best_params': gb_gs.best_params_
}

print('результаты моделей (log шкала)')
for name, metrics in results.items():
    print(f'\n{name}')
    print(f'  R2 {metrics["R2"]:.4f}')
    print(f'  MAE {metrics["MAE"]:.4f}')
    print(f'  RMSE {metrics["RMSE"]:.4f}')
    if 'best_params' in metrics:
        print(f'  лучшие параметры {metrics["best_params"]}')

print('\nкросс-валидация (5-fold)')
best_models = {
    'Ridge': Ridge(alpha=results['Ridge']['best_params']['alpha']),
    'Lasso': Lasso(alpha=results['Lasso']['best_params']['alpha'], max_iter=5000),
    'RandomForest': RandomForestRegressor(**results['RandomForest']['best_params'], random_state=42),
    'GradientBoosting': GradientBoostingRegressor(**results['GradientBoosting']['best_params'], random_state=42)
}
for name, model in best_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f'{name} R2 (cv) {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

best_gb = GradientBoostingRegressor(**results['GradientBoosting']['best_params'], random_state=42)
best_gb.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_gb.feature_importances_
}).sort_values('importance', ascending=False)

print('\nтоп важных признаков (GradientBoosting)')
for i, row in feature_importance.head(15).iterrows():
    print(f'{row["feature"]} {row["importance"]:.4f}')

# метрики в исходной шкале
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(best_gb.predict(X_test_scaled))

print('\nметрики в исходной шкале (SI)')
print(f'R2 {r2_score(y_test_orig, y_pred_orig):.4f}')
print(f'MAE {mean_absolute_error(y_test_orig, y_pred_orig):.2f}')
print(f'RMSE {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.2f}')