import ee
import pandas as pd
import os
import datetime
from datetime import datetime



def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def preprocess_drone_raw(drone_filename):
    df = pd.read_excel(drone_filename, sheet_name='샘플링위치별_식생지수및수확량')
    df['plot'] = df['ID'].apply(lambda x: ((x - 1) // 10) + 1)
    df['rep'] = df.groupby('plot').cumcount() + 1
    df = df.groupby('plot').mean().reset_index()
    df = df.drop(columns=['ID', 'yield(kg/10a)', 'rep'])

    return df

def reshape_df(df, key):
    melted_df = df.melt(id_vars=[key], var_name='VI', value_name='value')
    melted_df[['Date', 'VI']] = melted_df['VI'].str.split('_', expand=True)
    reshaped_df = melted_df.pivot_table(index=[key, 'Date'], columns='VI', values='value').reset_index()
    reshaped_df['Site'] = reshaped_df[key].apply(lambda x: f'{key}{x}')
    reshaped_df = reshaped_df.drop(columns=[key])
    return reshaped_df

def save_treatment_grouped(drone_filename, output_filename, treatment_dct, date_dct):
    df = preprocess_drone_raw(drone_filename)

    df_all = []

    for key, value in treatment_dct.items():
        mapping = {plot: treatment for treatment, plots in value.items() for plot in plots}
        df[key] = df['plot'].map(mapping)
        df_grouped = df.groupby(key).mean().reset_index().drop(columns=['plot'])
        df_treatment = reshape_df(df_grouped, key)
        df_all.append(df_treatment)
        del df[key]

    df_raw = reshape_df(df, 'plot')
    df_all.append(df_raw)

    df_all = pd.concat(df_all)
    df_all['Step'] = df_all['Date'].map(date_dct)
    df_all['Date'] = df_all['Date'].apply(lambda x: datetime.strptime(x, '%y%m%d').strftime('%Y-%m-%d'))
    df_all['Satellite'] = 'drone'

    df_all.to_csv(output_filename, index=False, encoding='utf-8-sig')

    return df_all
def main():
    input_dir = '../input'
    output_dir = make_dir('../output')
    save_dir = os.path.join(output_dir, 'eda_process')
    os.makedirs(save_dir, exist_ok=True)



    drone_filename = os.path.join(input_dir, '샘플링위치별_식생지수및수확량.xlsx')
    output_filename = os.path.join(save_dir, 'drone.csv')

    date_dct = {'230130': '분얼전','230321':'분얼전기', '230417':'분얼후기', '230501':'개화기', '230520':'개화2주후', '230601':'개화4주후',  '230615':'수확'}

    treatment_dct = {'seed': {'X': list(range(1, 5)), 'O': list(range(5, 9))},
                      'fertilizer': {'X': [1, 4, 5, 8], 'O': [2, 3, 6, 7]},
                      'water': {'X': [3, 4, 7, 8], 'O': [1, 2, 5, 6]}}

    save_treatment_grouped(drone_filename, output_filename, treatment_dct, date_dct)


if __name__ == '__main__':
    main()