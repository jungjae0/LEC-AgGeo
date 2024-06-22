import os
import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import platform

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

# satellite_colors = {'drone': 'green', 'sentinel_raw': 'blue', 'landsat8': 'red', 'landsat9': 'purple', 'sentinel_cloud_pre':'purple'}
satellite_colors = {'drone': 'green',
                    'sentinel2-cloud': 'blue', 'sentinel2-raw': 'blue',
                    'landsat8-cloud': 'red', 'landsat8-raw': 'red',
                    'landsat9-cloud': 'purple', 'landsat9-raw': 'purple'}

kosis_colors = {'미곡': 'orange', '잡곡': 'blue', '맥류': 'red', '두류': 'green'}


def preprocess_satellite(data_dir):
    drone_filename = os.path.join(data_dir, 'drone.csv')
    satellite_cloud_filename = os.path.join(data_dir, 'satellite_cloud.csv')
    satellite_raw_filename = os.path.join(data_dir, 'satellite_raw.csv')
    df_drone = pd.read_csv(drone_filename)
    # df_drone[df_drone.filter(like='RVI').columns] /= 10
    df_drone[df_drone.filter(like='CVI').columns] *= 10
    df_satellite_raw = pd.read_csv(satellite_raw_filename)
    df_satellite_raw['Satellite'] = df_satellite_raw['Satellite'] + '-raw'
    df_satellite_cloud = pd.read_csv(satellite_cloud_filename)
    df_satellite_cloud['Satellite'] = df_satellite_cloud['Satellite'] + '-cloud'
    df_satellite = pd.concat([df_satellite_raw, df_satellite_cloud])
    # df_satellite[df_satellite.filter(like='RVI').columns] /= 10
    df_satellite[df_satellite.filter(like='CVI').columns] /= 10
    df = pd.concat([df_drone, df_satellite])
    df[df.filter(like='RVI').columns] /= 10

    df_plot_avg = df[df['Site'].str.contains('seed')].drop(columns=['Site'])
    df_plot_avg = df_plot_avg.groupby(['Step', 'Satellite', 'Date']).mean().reset_index()
    df_plot_avg['Site'] = 'all'

    df = df[~df['Site'].str.contains('seed')]
    df = pd.concat([df, df_plot_avg])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # df.to_csv(output_df_filename, index=False, encoding='utf-8-sig')

    return df


def fig_satellite(df, vi_lst, save_path):
    df = df[df['Site'] == 'all']

    # nrows, ncols = 2, 3
    nrows, ncols = 1, 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    handles, labels = [], []
    df_raw = df[df['Satellite'].str.contains('raw')]
    df_cloud = df[df['Satellite'].str.contains('cloud')]
    drone = df[df['Satellite'] == 'drone']

    for i, vi in enumerate(vi_lst):
        # row, col = divmod(i, ncols)  # Calculate row and column index
        # ax = axes[row, col]
        ax = axes[i]
        ax = sns.lineplot(ax=ax, x="Step", y=vi, hue="Satellite",
                          palette=satellite_colors,
                          linestyle="--",
                          marker="o",
                          data=df_raw)

        ax = sns.lineplot(ax=ax, x="Step", y=vi, hue="Satellite",
                          palette=satellite_colors,
                          marker="o",
                          data=df_cloud)
        ax = sns.lineplot(ax=ax, x="Step", y=vi, hue="Satellite",
                          palette=satellite_colors,
                          marker="o",
                          data=drone)

        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.set_xlabel('Step', size=12)
        ax.set_ylabel(vi, size=12)
        ax.set_title(f'{vi}', size=13)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend_.remove()


    fig.legend(handles, labels, loc='upper right', fontsize=10)
    # plt.legend(title='Satellite')

    fig.suptitle('Tendency of Vegetation Index', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)

def preprocess_kosis(kosis_dir):
    kosis_files = [file for file in os.listdir(kosis_dir) if file.endswith('_정곡.csv')]

    df = pd.DataFrame()
    for kosis_file in kosis_files:
        # 1. 파일명에서 지역명 추출
        nm = kosis_file.rsplit('_', 1)[0]
        kosis_file = os.path.join(kosis_dir, kosis_file)
        df_each = pd.read_csv(kosis_file, encoding='cp949')

        # 2. 데이터 전처리
        # 2.1. 불필요한 열 제거 & 열 이름 변경
        df_each = df_each.drop([col for col in df_each.columns if 'Unnamed' in col], axis=1)
        c = [col for col in df_each.columns if '별' in col]
        for i in c:
            if '읍면' in i or '행정구역' in i:
                df_each = df_each.rename(columns={i: '읍면별'})
            else:
                df_each = df_each.rename(columns={i: '구분'})
        if '읍면별' in df_each.columns:
            df_each['행정구역'] = nm + '_' + df_each['읍면별']
        else:
            df_each['행정구역'] = '합계'

        df_each = df_each[df_each['행정구역'].str.contains('합계')]

        # 2.2. 행정구역 열 추가
        df = pd.concat([df, df_each])

    year_col = [f'{i} 년' for i in range(2014, 2022)]
    for i in year_col:
        if i not in df.columns:
            df[i] = '-'
    df = df.fillna('-')

    # 2.3. 필요없는 열 drop & 값 대체
    id_vars = ['행정구역', '구분', '항목', '단위']
    df = df[id_vars + year_col]
    df = df.replace(0, '-')
    df['항목'] = df['항목'].apply(lambda row: re.findall('[가-힣]+', row)[0])

    # 3. 데이터 구조 변경
    # 3.1. '연도' 열을 갖도록 정리 & 필요없는 행 제거
    df = df.melt(id_vars=id_vars, var_name='year', value_name='값')
    df = df.fillna('-')
    df = df[df['값'] != '-']
    df['year'] = df['year'].apply(lambda row: re.findall(r'\d+', row)[0]).astype(int)
    df['행정구역'] = df['행정구역'].apply(lambda x: x.split('_')[0] + ' ' + x.split('_')[1])
    df = df[df['항목'] == '생산량']
    df['값'] = df['값'].astype(float)
    df = df[(df['구분'] != '합계') & (df['구분'] != '서류')]

    return df

def fig_kosis(df, save_path):
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    handles, labels = [], []


    geo = df['행정구역'].unique()
    for i, t in enumerate(geo):
        ax = axes[i]
        df_t = df[df['행정구역'] == t]
        ax = sns.lineplot(ax=ax, x='year', y='값',
                          hue='구분', palette=kosis_colors,
                          marker="o",
                          data=df_t)

        ax.set_title(f'{t.replace("전라북도", "전북특별자치도")}', size=15)
        ax.set_xlabel('Year', size=13)
        ax.set_ylabel('Production(M/T)', size=12)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend_.remove()


    fig.legend(handles, labels, loc='upper right', fontsize=10)
    # plt.legend(title='구분')
    fig.suptitle('Yearly Production', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)

def preprocess_weather(weather_dir, codes):
    os.makedirs(weather_dir, exist_ok=True)

    df = pd.DataFrame()
    for sig, code in codes.items():
        each_save_path = os.path.join(weather_dir, f'{sig}.csv')
        if not os.path.exists(each_save_path):
            print('Downloading weather data for', sig)
            url = f"https://api.taegon.kr/stations/{code}/?sy=2014&ey=2022&format=csv"
            df_each = pd.read_csv(url, sep='\\s*,\\s*', engine="python")
            df_each.columns = [col.strip() for col in df_each.columns]
            df_each['행정구역'] = sig
            df_each.to_csv(os.path.join(weather_dir, f'{sig}.csv'), index=False, encoding='utf-8-sig')

        else:
            df_each = pd.read_csv(each_save_path)

        df_each['month'] = df_each['month'].apply(lambda num: '{:02d}'.format(num))
        df_each = df_each.groupby(['year', 'month']).agg({'tavg': 'mean',
                                                          'rainfall': 'sum',
                                                          'humid': 'mean',
                                                          'tmax': 'max',
                                                          'tmin': 'min'}).reset_index()
        # df_each = df_each.pivot(index='year', columns='month', values=['tavg', 'rainfall', 'humid', 'tmax', 'tmin'])
        # df_each.columns = ['_'.join(col) for col in df_each.columns]
        df_each['시군구'] = sig

        df = pd.concat([df, df_each])

    return df

def fig_weather_tavg(df, save_path):
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    handles, labels = [], []

    geo = df['시군구'].unique()
    for i, t in enumerate(geo):
        ax = axes[i]
        df_t = df[df['시군구'] == t]
        ax = sns.lineplot(ax=ax, x='month', y='tavg', hue='year',
                          marker="o",
                          data=df_t)

        if t == '해남군':
            t = '전라남도 해남군'
        else:
            t = '전북특별자치도 ' + t

        ax.set_title(f'{t}', size=13)
        ax.set_xlabel('Month', size=12)
        ax.set_ylabel('Temperature(°C)', size=12)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend_.remove()

    # fig.legend(handles, labels, loc='upper right', fontsize=10)
    plt.suptitle('Monthly Average Temperature by Year', size=20, fontweight='bold')
    plt.legend(title='Year')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)

def fig_weather_rainfall(df, save_path):
    nrows, ncols = 3,1


    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 15, nrows * 5))

    handles, labels = [], []

    geo = df['시군구'].unique()
    for i, t in enumerate(geo):
        ax = axes[i]
        df_t = df[df['시군구'] == t]
        ax = sns.barplot(ax=ax, data=df_t, x='month', y='rainfall', hue='year')
        if t == '해남군':
            t = '전라남도 해남군'
        else:
            t = '전북특별자치도 ' + t

        ax.set_title(f'{t}', size=15)
        ax.set_xlabel('Month', size=12)
        ax.set_ylabel('Rainfall (mm)', size=12)

        # if i == 0:
        #     handles, labels = ax.get_legend_handles_labels()

        # ax.legend_.remove()
    plt.suptitle('Monthly Rainfall by Year\n', size=20, fontweight='bold')
    plt.tight_layout()
    plt.legend(title='Year')

    # plt.show()
    plt.savefig(save_path)


def main():
    output_dir = '../output'
    kosis_dir = '../input/통계청_생산량'
    eda_dir = os.path.join(output_dir, 'eda_process')
    save_dir = make_dir(os.path.join(output_dir, 'eda_results'))


    vi_lst = ['NDVI', 'CVI', 'NDRE', 'GNDVI', 'RVI']

    df_satellite = preprocess_satellite(eda_dir)
    satellite_save_path= os.path.join(save_dir, 'satellite.png')
    fig_satellite(df_satellite, vi_lst, satellite_save_path)

    df_kosis = preprocess_kosis(kosis_dir)
    kosis_save_path= os.path.join(save_dir, 'kosis.png')
    fig_kosis(df_kosis, kosis_save_path)


    codes = {'해남군': '261', '부안군': '243', '김제시': '243'}
    weather_dir = os.path.join(output_dir, 'weather')
    df_weather = preprocess_weather(weather_dir, codes)
    # print(df_weather)
    weather_save_path = os.path.join(save_dir, 'weather_tavg.png')
    fig_weather_tavg(df_weather, weather_save_path)

    weather_save_path = os.path.join(save_dir, 'weather_rainfall.png')
    fig_weather_rainfall(df_weather, weather_save_path)





if __name__ == '__main__':
    main()
