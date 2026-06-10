import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# таргет CC50 > медианы
median_cc50 = df['CC50, mM'].median()
y = (df['CC50, mM'] > median_cc50).astype(int)

print(f'медиана CC50 {median_cc50:.2f}')
print(f'распределение классов 0 {(y==0).sum()} 1 {(y==1).sum()}')

# признаки
exclude_cols = ['IC50, mM', 'CC50, mM', 'SI']
X = df.drop(exclude_cols, axis=1)

#сплит
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# словарь результатов
results = {}

# логистическая регрессия
lr_params = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
lr = LogisticRegression(max_iter=5000, random_state=42)
lr_gs = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc')
lr_gs.fit(X_train_scaled, y_train)
y_pred_lr = lr_gs.predict(X_test_scaled)
y_prob_lr = lr_gs.predict_proba(X_test_scaled)[:, 1]
results['LogisticRegression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_prob_lr),
    'best_params': lr_gs.best_params_
}

# random forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
rf = RandomForestClassifier(random_state=42)
rf_gs = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
rf_gs.fit(X_train_scaled, y_train)
y_pred_rf = rf_gs.predict(X_test_scaled)
y_prob_rf = rf_gs.predict_proba(X_test_scaled)[:, 1]
results['RandomForest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_prob_rf),
    'best_params': rf_gs.best_params_
}

# gradient boosting
gb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
gb = GradientBoostingClassifier(random_state=42)
gb_gs = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc')
gb_gs.fit(X_train_scaled, y_train)
y_pred_gb = gb_gs.predict(X_test_scaled)
y_prob_gb = gb_gs.predict_proba(X_test_scaled)[:, 1]
results['GradientBoosting'] = {
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'precision': precision_score(y_test, y_pred_gb),
    'recall': recall_score(y_test, y_pred_gb),
    'f1': f1_score(y_test, y_pred_gb),
    'roc_auc': roc_auc_score(y_test, y_prob_gb),
    'best_params': gb_gs.best_params_
}

# SVC
svc_params = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
svc = SVC(probability=True, random_state=42)
svc_gs = GridSearchCV(svc, svc_params, cv=5, scoring='roc_auc')
svc_gs.fit(X_train_scaled, y_train)
y_pred_svc = svc_gs.predict(X_test_scaled)
y_prob_svc = svc_gs.predict_proba(X_test_scaled)[:, 1]
results['SVC'] = {
    'accuracy': accuracy_score(y_test, y_pred_svc),
    'precision': precision_score(y_test, y_pred_svc),
    'recall': recall_score(y_test, y_pred_svc),
    'f1': f1_score(y_test, y_pred_svc),
    'roc_auc': roc_auc_score(y_test, y_prob_svc),
    'best_params': svc_gs.best_params_
}

print('\nрезультаты моделей')
for name, metrics in results.items():
    print(f'\n{name}')
    print(f'accuracy {metrics["accuracy"]:.4f}')
    print(f'precision {metrics["precision"]:.4f}')
    print(f'recall {metrics["recall"]:.4f}')
    print(f'f1 {metrics["f1"]:.4f}')
    print(f'roc_auc {metrics["roc_auc"]:.4f}')
    print(f'лучшие параметры {metrics["best_params"]}')

print('\nкросс-валидация (5-fold)')
best_models = {}
for name in results:
    if name == 'LogisticRegression':
        best_models[name] = LogisticRegression(**results[name]['best_params'], max_iter=5000, random_state=42)
    elif name == 'RandomForest':
        best_models[name] = RandomForestClassifier(**results[name]['best_params'], random_state=42)
    elif name == 'GradientBoosting':
        best_models[name] = GradientBoostingClassifier(**results[name]['best_params'], random_state=42)
    elif name == 'SVC':
        best_models[name] = SVC(**results[name]['best_params'], probability=True, random_state=42)

for name, model in best_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f'{name} ROC-AUC (cv) {cv_scores.mean():.4f} +- {cv_scores.std():.4f}')