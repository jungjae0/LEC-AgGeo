import json
import re

import pandas as pd
import os

def preprocess_satellite(csv_dir, save_dir):
    lst_filename = os.listdir(csv_dir)

    df = pd.DataFrame()
    for filename in lst_filename:
        filepath = os.path.join(csv_dir, filename)
        filesize = os.path.getsize(filepath)
        # 1.1. df.empty and df.all = 0 인 파일은 제외
        if not filesize < 20:
            df_each = pd.read_csv(filepath)
            if list(df_each.iloc[:, 0].unique())[0] == 0:
                continue
            else:
                # 1.2. 읍면동 열 추가
                df_each['행정구역'] = filename.replace('.csv', '')
                df = pd.concat([df, df_each])

    # 2. 데이터 전처리
    # 2.1. 모든 열이 0인 행 제외
    lst_vi_col = [col for col in df.columns if '_' in col]
    df = df.loc[~(df[lst_vi_col].eq(0).any(axis=1))]
    df[df.filter(like='RVI').columns] /= 10
    df[df.filter(like='CVI').columns] /= 10


    # 2.2. day를 A, B, C로 변경
    df['date'] = df['date'].astype(str).str.replace('.zip', '')
    df['year'] = df['date'].str[:4]
    df['month'] = df['date'].str[4:6]
    df = df[(df['month'].astype(int) > 2) & (df['month'].astype(int) < 11)]
    df['yearmonth'] = df['year'] + df['month']
    df['day'] = df['date'].str[6:]
    df['seq'] = df.groupby(['yearmonth', '행정구역']).cumcount() + 1
    df['seq'] = df['seq'].apply(lambda n: chr(64 + n))
    df['seq'] = df['month'] + df['seq']
    df = df.drop(columns=['yearmonth', 'date', 'month', 'day'])

    # 3. 데이터 구조 변경
    # 3.1. 'vi_desc_year_day'열을 갖도록 정리
    df = df.pivot(index=['행정구역', 'year'], columns='seq', values=lst_vi_col)
    df.columns = ['_'.join(col) for col in df.columns]
    df = df.reset_index()
    # df = df.drop(columns=[col for col in df.columns if 'min' in col])
    df['year'] = df['year'].astype(int)

    ## -- 결과: year, 읍면동, '{vi}_{desc}_{year}{day}'로 구성된 데이터프레임
    # 4. 데이터 저장
    save_path = os.path.join(save_dir, '위성영상_식생지수.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    return df


def preprocess_kosis(kosis_dir, save_dir):
    kosis_files = os.listdir(kosis_dir)
    kosis_files = [f for f in kosis_files if f.endswith('.csv')]
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

        # 2.2. 행정구역 열 추가
        df = pd.concat([df, df_each])

    year_col = [f'{i} 년' for i in range(2014, 2022)]
    for i in year_col:
        if i not in df.columns:
            df[i] = '-'
    df = df.fillna('-')

    # 2.3. 필요없는 열 drop & 값 대체
    id_vars= ['행정구역', '구분', '항목', '단위']
    df = df[id_vars + year_col]
    df = df.replace(0, '-')
    df['항목'] = df['항목'].apply(lambda row: re.findall('[가-힣]+', row)[0])

    # 3. 데이터 구조 변경
    # 3.1. '연도' 열을 갖도록 정리 & 필요없는 행 제거
    df = df.melt(id_vars=id_vars, var_name='year', value_name='값')
    df = df.fillna('-')
    df = df[df['값'] != '-']
    df['year'] = df['year'].apply(lambda row: re.findall(r'\d+', row)[0]).astype(int)


    # -- 결과: 행정구역, 구분, 항목, 단위, 연도 로 구성된 데이터프레임
    save_path = os.path.join(save_dir, '통계청_전체_생산량.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    return df

def preprocess_weather(weather_dir, save_dir, codes):
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
        df_each = df_each.pivot(index='year', columns='month', values=['tavg', 'rainfall', 'humid', 'tmax', 'tmin'])
        df_each.columns = ['_'.join(col) for col in df_each.columns]
        df_each['시군구'] = sig

        df = pd.concat([df, df_each])

    df = df.reset_index()
    save_path = os.path.join(save_dir, f'기상정보.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    return df


def main():
    select_sig = ['해남군','김제시','부안군']
    codes = {'해남군': '261', '부안군': '243', '김제시': '243'}

    output_dir = './output'
    save_dir = os.path.join(output_dir, 'model_process')
    os.makedirs(output_dir, exist_ok=True)
    input_dir = './input'
    csv_dir = os.path.join(output_dir, 'vi_avg')
    kosis_dir = os.path.join(input_dir, '통계청_생산량')
    weather_dir = os.path.join(output_dir, 'weather')

    # 1. 위성 데이터 전처리
    satellite = preprocess_satellite(csv_dir, save_dir)

    # 2. 통계청 데이터 전처리
    kosis = preprocess_kosis(kosis_dir, save_dir)

    # 3. 기상청 데이터 전처리
    weather = preprocess_weather(weather_dir, save_dir, codes)

    # 4. 데이터 병합
    satellite_kosis = pd.merge(satellite, kosis, on=['year', '행정구역'], how='inner')
    satellite_kosis['시군구'] = satellite_kosis['행정구역'].apply(lambda row: row.split('_')[1])
    satellite_kosis = pd.merge(satellite_kosis, weather, on=['year', '시군구'], how='inner')
    satellite_kosis.to_csv(os.path.join(save_dir, '전체_데이터.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
