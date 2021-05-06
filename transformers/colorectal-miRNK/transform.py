import numpy as np
import pandas as pd


BASE_DIR = 'data/colorectal-miRNK'

# DATA
data = pd.read_csv(f'{BASE_DIR}/secondary/isomiR_CPM.tsv', index_col=0, sep='\t').transpose()
samples = pd.read_csv(f'{BASE_DIR}/secondary/tumor_samples.tsv', index_col=False, sep='\t')

#  оставляем только данные, которые есть в samples
data = data.loc[data.index.isin(samples['Tumor ID'])]

# из айдишников клиентов убираем лишнее окончание
data.index = map(lambda name: '-'.join(name.split('-')[:-1]), data.index)

# CLINICAL
clinical = pd.read_csv(f'{BASE_DIR}/secondary/clinical.tsv', index_col=False, sep='\t')[[
    'case_submitter_id',
    'vital_status',
    'days_to_death',
    'days_to_last_follow_up',
    'ajcc_pathologic_stage',
]]

# заменяем плохое на нули
clinical['days_to_death'] = clinical['days_to_death']\
    .replace("'--", '0')\
    .replace(np.nan, '0')\
    .astype(float)

clinical['days_to_last_follow_up'] = clinical['days_to_last_follow_up']\
    .replace("'--", '0')\
    .replace(np.nan, '0')\
    .astype(float)

clinical['ajcc_pathologic_stage'] = clinical['ajcc_pathologic_stage']\
    .replace("'--", '')

# сортируем по времени прихода
clinical = clinical.sort_values(
    by=['days_to_last_follow_up', 'days_to_death'],
    ascending=[False, True],
)

# оставляем только тех, кто есть в данных микроРНК
clinical = clinical[clinical['case_submitter_id'].isin(data.index)]

# оставляем только тех у кого не оба нуля в данных
clinical = clinical[~((clinical['days_to_death'] == 0) & (clinical['days_to_last_follow_up'] == 0))]

# удаляем дубликаты по айди клиента
clinical = clinical.drop_duplicates(subset='case_submitter_id')
clinical = clinical.set_index('case_submitter_id')
clinical.index.name = None

# убираем тех, кто мертв, но days_to_death равен нулю
clinical = clinical[~((clinical['vital_status'] == 'Dead') & (clinical['days_to_death'] == 0))]

# бинаризируем класс
clinical['vital_status'] = clinical['vital_status'].apply(lambda x: 1 if x == 'Dead' else 0)

# соединяем колонки времени, в приоритете days_to_death
clinical['Time to event'] = clinical.apply(lambda row: row['days_to_death'] or row['days_to_last_follow_up'], axis=1)

# удаляем старые колонки времени
clinical = clinical.drop(['days_to_death', 'days_to_last_follow_up'], axis=1)

# переименовываем колонки
clinical = clinical.rename(columns={
    'vital_status': 'Event',
    'ajcc_pathologic_stage': 'Stage',
})

# сортируем по айди клиента для дальнейшего удобства
clinical = clinical.sort_index()

print(clinical)
clinical.to_csv(f'{BASE_DIR}/annotation.csv')


data = data.loc[data.index.isin(clinical.index)]
# сортируем по айди клиента для дальнейшего удобства
data = data.sort_index()

# неотфильтрованные данные
print(data)
data.to_csv(f'{BASE_DIR}/data.csv')

# фильтрация: оставляем только микроРНК с "большой"(топ 20%) медианой
data = data.drop(
    columns=data.columns[data.median() <= data.median().quantile(q=0.8)],
    axis=1,
)
print(data)
data.to_csv(f'{BASE_DIR}/data_filtered.csv')
