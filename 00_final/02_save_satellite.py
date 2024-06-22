import json
import shutil
import ee
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from ast import literal_eval
import requests
import os

ee.Authenticate()

ee.Initialize(project='ee-jyihaan4')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def apply_scale_factors(image, img_key):
    # 1. 위성영상에 따라 스케일링 팩터 적용
    if 'landsat' in img_key:
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)
    elif 'sentinel' in img_key:
        return image.divide(10000)


def mask_clouds(image, img_key):
    # 1. 위성영상에 따라 클라우드 마스킹
    if 'sentinel' in img_key:
        qa = image.select('QA60')

        # 1.1. 구름과 권운 비트 마스크
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # 1.2. 마스킹 조건 정의

        mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
                .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))

        return image.updateMask(mask)

    elif 'landsat' in img_key:
        qa = image.select('QA_PIXEL')

        # 1.1. 구름과 그림자 비트 마스크
        cloud_shadow_bit_mask = 1 << 3
        clouds_bit_mask = 1 << 5

        # 1.2. 마스킹 조건 정의
        mask = (qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0)
                .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)))

        return image.updateMask(mask)

def add_indices_band(image, img_key):
    if 'landsat' in img_key:
        band_dct = {'NIR': 'SR_B5', 'RED': 'SR_B4', 'GREEN': 'SR_B3', 'RED_EDGE': 'SR_B6', 'BLUE': 'SR_B2'}

    elif 'sentinel' in img_key:
        band_dct = {'NIR': 'B8', 'RED': 'B4', 'GREEN': 'B3', 'RED_EDGE': 'B5', 'BLUE': 'B2', 'SWIR': 'B11'}

    NIR = image.select(band_dct['NIR'])
    RED = image.select(band_dct['RED'])
    GREEN = image.select(band_dct['GREEN'])

    # 1. 식생지수 산출

    # 1.1. NDVI = (NIR - RED) / (NIR + RED)
    ndvi = image.normalizedDifference([band_dct['NIR'], band_dct['RED']]).rename('NDVI')

    # 1.2. NDRE = (NIR - RED_EDGE) / (NIR + RED_EDGE)
    ndre = image.normalizedDifference([band_dct['NIR'], band_dct['RED_EDGE']]).rename('NDRE')

    # 1.3. GNDVI = (NIR - GREEN) / (NIR + GREEN)
    gndvi = image.normalizedDifference([band_dct['NIR'], band_dct['GREEN']]).rename('GNDVI')

    # 1.4. RVI = NIR / RED
    rvi = NIR.divide(RED).rename('RVI')

    # 1.5. CVI = (RED / GREEN^2) * NIR
    cvi = image.expression(
        '(RED / GREEN ** 2) * NIR', {
            'NIR': image.select(band_dct['NIR']),
            'GREEN': image.select(band_dct['GREEN']),
            'RED': image.select(band_dct['RED'])
        }).rename('CVI')

    # 2. 위성영상에 식생지수 밴드 추가
    return image.addBands([ndvi, cvi, ndre, gndvi, rvi])


def get_satellite(start_date, end_date, site, img_key, imgs):
    if 'sentinel' in img_key:
        cloud_percentage_band = 'CLOUDY_PIXEL_PERCENTAGE'
    elif 'landsat' in img_key:
        cloud_percentage_band = 'CLOUD_COVER'

    # 1. 날짜, 지역
    filtered_imgs = imgs.filterDate(start_date, end_date).filterBounds(site)


    # 2. 클라우드 마스킹, 스케일링 팩터 적용, 식생지수 밴드 추가
    processed_img_collection = (filtered_imgs.map(lambda image: mask_clouds(image, img_key))
                                             .map(lambda image: apply_scale_factors(image, img_key))
                                             .map(lambda image: add_indices_band(image, img_key)))


    return processed_img_collection


def save_satellite(collection, geometry, adm_nm, check_point, img_key, output_dir):
    local_folder = os.path.join(output_dir, rf'satellite_img_{check_point}/{adm_nm}')
    os.makedirs(local_folder, exist_ok=True)

    images = collection.sort('system:time_start').toList(collection.size())

    for i in tqdm(range(images.size().getInfo()), desc=f'{adm_nm}'):

        # 1. 이미지 컬렉션에서 하나의 이미지 추출 & 날짜 정보 추출
        image = ee.Image(images.get(i))

        if img_key == 'sentinel2':
            date = image.getInfo()['properties']['system:index'][:8]
        else:
            date = image.getInfo()['properties']['system:index'].split('_')[-1]

        # 2. 다운로드할 밴드 선택 & 이미지 다운로드
        bands = ['NDVI', 'CVI', 'NDRE', 'GNDVI', 'RVI']

        file_path = os.path.join(local_folder, f'Image_{date}.zip')
        if not os.path.exists(file_path):
            # 2.1. 해상도 = 30m, 좌표계 = WGS84
            url = image.select(bands).getDownloadURL({
                'scale': 30,
                'crs': 'EPSG:4326',
                'region': geometry
            })

            response = requests.get(url, stream=True)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            pass
def process_row(row, images_dct, output_dir):
    adm_nm = row['adm_nm']
    geometry = ee.Geometry.Polygon(row['coordinates'][0])

    for img_key, values in images_dct.items():
        imgs, start_date, end_date = values['imgs'], values['start_date'], values['end_date']

        for check_point in ['raw']:
            img_col = get_satellite(start_date, end_date, geometry, img_key, imgs)
            save_satellite(img_col, geometry, adm_nm, check_point, img_key, output_dir)


def main():
    input_dir = make_dir('./input')
    output_dir = rf'Z:\users\jaeyoung'

    json_file_path = os.path.join(input_dir, 'hangjeongdong.json')
    with open(json_file_path, encoding='utf-8') as f:
        geojson = json.load(f)

    df = pd.json_normalize(geojson['features'])
    df.columns = [col.split('.')[-1] for col in df.columns]
    df = df[df['sggnm'].isin(['김제시', '부안군', '해남군'])]
    df['adm_nm'] = df['adm_nm'].str.replace(' ', '_')

    images_dct = {
        'landsat8': {'imgs': ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), 'start_date': '2014-01-01', 'end_date': '2018-12-31'},
        'sentinel2': {'imgs': ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"), 'start_date': '2019-01-01', 'end_date': '2021-12-31'}
                  }

    df.apply(lambda row: process_row(row, images_dct, output_dir), axis=1)

if __name__ == '__main__':
    main()