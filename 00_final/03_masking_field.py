import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import tqdm
import time
import json
import pandas as pd
import warnings
import zipfile


warnings.filterwarnings("ignore", category=RuntimeWarning)


def process_zip(root_dir, img_dir):
    # 1. 일단위 식생지수 tif 압축 폴더 & 압축해제 폴더 경로 설정
    zip_file_path = os.path.join(root_dir, img_dir)
    extract_to_path = os.path.join(root_dir, img_dir.replace('.zip', ''))
    if not os.path.exists(extract_to_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
    return extract_to_path

def process_vi(extract_to_path, geom_json):

    # 1. 식생지수별 임의의 임계값 설정
    vi_dct = {'NDVI': 0.2, 'CVI': 2, 'NDRE': 0.2, 'GNDVI': 0.2, 'RVI': 2}

    vi_value = {}
    for vi, threshold in vi_dct.items():
        tif_path = [file for file in os.listdir(extract_to_path) if vi in file][0]
        vi_path = os.path.join(extract_to_path, tif_path)
        vi_dataset = rasterio.open(vi_path)
        out_image, out_transform = mask(vi_dataset, geom_json, crop=True)
        masked_vi = out_image[0]

        # 2. 임계값 이상인 값들만 추출
        valid_vi = masked_vi[masked_vi != vi_dataset.nodata]
        valid_vi = valid_vi[valid_vi >= threshold]

        # 3. 평균, 최대, 최소, 중앙값 계산
        if valid_vi.size > 0:

            mean_vi = valid_vi.mean()
            max_vi = valid_vi.max()
            min_vi = valid_vi.min()
            median_vi = np.median(valid_vi)
        else:
            mean_vi = 0
            max_vi = 0
            min_vi = 0
            median_vi = 0

        vi_value[f'{vi}_mean'] = mean_vi
        vi_value[f'{vi}_max'] = max_vi
        vi_value[f'{vi}_min'] = min_vi
        vi_value[f'{vi}_median'] = median_vi

    return vi_value


def process_values(img_dir_lst, adm_nm, root_dir, geom_json):
    value_lst = []
    for img_dir in tqdm.tqdm(img_dir_lst, total=len(img_dir_lst), desc=f'{adm_nm}'):
        if img_dir.endswith('.zip'):
            try:
                # 1. 압축 폴더 압축 해제
                extract_to_path = process_zip(root_dir, img_dir)

                # 2. 일단위 식생지수 값 처리
                date = img_dir.replace('Image_', '').replace('.zip', '')

                vi_value_dct = process_vi(extract_to_path, geom_json)
                vi_value_dct['date'] = date

                value_lst.append(vi_value_dct)

            except:
                continue
    return value_lst

def field_avg(geom_json, adm_nm, d_root, avg_save_dir):
    os.makedirs(avg_save_dir, exist_ok=True)

    root_dir = os.path.join(d_root, f'{adm_nm}')
    img_dir_lst = os.listdir(root_dir)
    save_path = os.path.join(avg_save_dir, f'{adm_nm}.csv')

    # 1. 데이터프레임 형태로 정리 & csv 파일 저장

    if os.path.exists(save_path):
        img_dir_lst = [file for file in img_dir_lst if int(file.replace('Image_', '')[:4]) < 2019]

        df_exists = pd.read_csv(save_path)
        value_lst= process_values(img_dir_lst, adm_nm, root_dir, geom_json)
        df_new = pd.DataFrame(value_lst)
        df = pd.concat([df_exists, df_new])

    else:

        img_dir_lst = [file for file in img_dir_lst if int(file.replace('Image_', '')[:4]) < 2022]
        value_lst= process_values(img_dir_lst, adm_nm, root_dir, geom_json)
        df = pd.DataFrame(value_lst)

    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def get_field_geom(d_dir, nm, standard_crs):
    cover_path = os.path.join(d_dir, fr'{nm}\{nm}.shp')
    cover = gpd.read_file(cover_path)

    fields = cover[cover['L2_NAME'] == '논']
    if fields.crs != standard_crs:
        fields = fields.to_crs(standard_crs)

    all_geometries = fields.geometry.unary_union
    geom_json = [all_geometries.__geo_interface__]

    return geom_json


def main():
    output_dir = './output'
    d_dir = r'D:\Projects\202406_토지피복도병합'
    root_dir = r"Z:\users\jaeyoung\satellite_img_raw"

    save_dir = os.path.join(output_dir, f'vi_avg')

    standard_crs = 'EPSG:4326'

    adm_nm = ['김제시', '부안군', '해남군']

    for nm in adm_nm:
        geom_json = get_field_geom(d_dir, nm, standard_crs)

        ls = [d for d in os.listdir(root_dir) if '김제시' in d]

        for i in ls:
            field_avg(geom_json, i, root_dir, save_dir)

if __name__ == '__main__':
    main()