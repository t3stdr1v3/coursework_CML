import pandas as pd
import numpy as np

df = pd.read_excel('data/Данные_для_курсовои_Классическое_МО.xlsx')
print('размер данных до очистки', df.shape)

# удаляем служебную колонку
df = df.drop('Unnamed: 0', axis=1)

# проверяем пропуски
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_info = pd.DataFrame({
    'количество': missing[missing > 0],
    'процент': missing_pct[missing_pct > 0]
})
if len(missing_info) > 0:
    print('\nколонки с пропусками')
    print(missing_info)
else:
    print('\nпропусков нет')

# столбцы с одинаковым значениями
zero_var_cols = []
for col in df.columns:
    if df[col].nunique() <= 1:
        zero_var_cols.append(col)

print('\nстолбцы с одинаковыии значениями')
for col in zero_var_cols:
    print(f'  {col} значение {df[col].iloc[0]}')

# столбцы где больше 90% нулей
zero_dominant_cols = []
for col in df.columns:
    zero_pct = (df[col] == 0).sum() / len(df) * 100
    if zero_pct > 90:
        zero_dominant_cols.append((col, zero_pct))

print('\nстолбцы где больше 90% нулей')
for col, pct in sorted(zero_dominant_cols, key=lambda x: x[1], reverse=True):
    print(f'  {col} {pct:.1f}% нулей')

# дубликаты
duplicates = df.duplicated().sum()
print(f'\nдубликатов строк {duplicates}')

# строки с пропусками (инфа по пропускам собрана заранее)
cols_with_nan = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 
                 'MinAbsPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
                 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 
                 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']

nan_rows = df[df[cols_with_nan].isnull().any(axis=1)]
print(f'\nстрок с пропусками {len(nan_rows)}, индексы {nan_rows.index.tolist()}')
print('целевые переменные в этих строках')
print(nan_rows[['IC50, mM', 'CC50, mM', 'SI']])

# удаляем строки с пропусками
df = df.dropna()

# удаляем дубликаты
df = df.drop_duplicates()

# удаляем столбцы без вариативности
df = df.drop(zero_var_cols, axis=1)
print(f'\nразмер после очистки {df.shape}')

# таргеты
targets = ['IC50, mM', 'CC50, mM', 'SI']

print('\nстатистики таргетов')
print(df[targets].describe())

# проверка формуы SI = CC50/IC50
calculated_SI = df['CC50, mM'] / df['IC50, mM']
si_diff = abs(df['SI'] - calculated_SI)
print(f'\nмакс расхождение SI с CC50/IC50 {si_diff.max():.10f}')

# логарифмируем таргеты
for col in targets:
    log_col = 'log_' + col.replace(', ', '_').replace(' ', '_')
    df[log_col] = np.log1p(df[col])

log_targets = [col for col in df.columns if col.startswith('log_')]

print('\nстатистики после логарифмирования')
print(df[log_targets].describe())

# выбросы в логарифмированных данных
for col in log_targets:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'{col} выбросов {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)')

# выбросы в исходных данных
print('\nвыбросы в исходных данных (IQR)')
for col in targets:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'{col} выбросов {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)')

# корреляция признаков с целевыми переменными
feature_cols = [col for col in df.columns if col not in targets + log_targets]

print('\nтоп признаков по корреляции с IC50')
corr_ic50 = df[feature_cols].corrwith(df['IC50, mM']).abs().sort_values(ascending=False)
for feat, val in corr_ic50.head(15).items():
    print(f'  {feat} {val:.4f}')

print('\nтоп признаков по корреляции с CC50')
corr_cc50 = df[feature_cols].corrwith(df['CC50, mM']).abs().sort_values(ascending=False)
for feat, val in corr_cc50.head(15).items():
    print(f'  {feat} {val:.4f}')

print('\nтоп признаков по корреляции с SI')
corr_si = df[feature_cols].corrwith(df['SI']).abs().sort_values(ascending=False)
for feat, val in corr_si.head(15).items():
    print(f'  {feat} {val:.4f}')

# мультиколлинеарность
high_corr_pairs = []
corr_matrix = df[feature_cols].corr().abs()
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j]))

print(f'\nпар с корреляцией > 0.95 {len(high_corr_pairs)}')
print('первые 10 пар')
for feat1, feat2, val in high_corr_pairs[:10]:
    print(f'  {feat1} - {feat2} {val:.4f}')

print('\nкорреляция между целевыми переменными')
print(df[targets].corr())

print('\nкорреляция между логарифмами')
print(df[log_targets].corr())

# fr_ сильно разряженные признаки
fr_cols = [col for col in df.columns if col.startswith('fr_')]
print('\nfr_ признаки с <=10 единиц')
for col in fr_cols:
    ones = df[col].sum()
    if ones <= 10:
        print(f'  {col} {int(ones)} единиц из {len(df)}')

# доля SI > 8
print(f'\nSI > 8 {(df["SI"] > 8).sum()} из {len(df)} ({(df["SI"] > 8).mean()*100:.1f}%)')

# медианы для задач классификации
print(f'\nмедианы для классификации')
print(f'IC50 медиана {df["IC50, mM"].median():.2f}')
print(f'CC50 медиана {df["CC50, mM"].median():.2f}')
print(f'SI медиана {df["SI"].median():.2f}')