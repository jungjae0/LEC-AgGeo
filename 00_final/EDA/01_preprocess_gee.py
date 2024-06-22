import ee
import pandas as pd
import os
import json
import tqdm
from datetime import datetime

ee.Authenticate()

ee.Initialize(project='ee-jyihaan4')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def apply_scale_factors(image, img_key):
    if 'landsat' in img_key:
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)
    elif 'sentinel' in img_key:
        return image.divide(10000)


def mask_clouds(image, img_key):
    if 'sentinel' in img_key:
        qa = image.select('QA60')

        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
                .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))

        return image.updateMask(mask)

    elif 'landsat' in img_key:
        cloud_shadow_bit_mask = 1 << 3
        clouds_bit_mask = 1 << 5

        qa = image.select('QA_PIXEL')

        mask = (qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0)
                .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)))

        return image.updateMask(mask)

def resampling(image, img_key):
    target_scale = 0.03
    if 'landsat' in img_key:
        crs = image.select('SR_B2').projection().crs()
    elif 'sentinel' in img_key:
        crs = image.select('B2').projection().crs()

    return image.resample('bilinear').reproject(crs=crs, scale=target_scale)

def add_indices_band(image, img_key):
    if 'landsat' in img_key:
        band_dct = {'NIR': 'SR_B5', 'RED': 'SR_B4', 'GREEN': 'SR_B3', 'RED_EDGE': 'SR_B6', 'BLUE': 'SR_B2'}

    elif 'sentinel' in img_key:
        band_dct = {'NIR': 'B8', 'RED': 'B4', 'GREEN': 'B3', 'RED_EDGE': 'B5', 'BLUE': 'B2'}

    NIR = image.select(band_dct['NIR'])
    RED = image.select(band_dct['RED'])
    GREEN = image.select(band_dct['GREEN'])

    ndvi = image.normalizedDifference([band_dct['NIR'], band_dct['RED']]).rename('NDVI')
    ndre = image.normalizedDifference([band_dct['NIR'], band_dct['RED_EDGE']]).rename('NDRE')
    gndvi = image.normalizedDifference([band_dct['NIR'], band_dct['GREEN']]).rename('GNDVI')
    rvi = NIR.divide(RED).rename('RVI')
    cvi = image.expression(
        '(RED / GREEN ** 2) * NIR', {
            'NIR': image.select(band_dct['NIR']),
            'GREEN': image.select(band_dct['GREEN']),
            'RED': image.select(band_dct['RED'])
        }).rename('CVI')
    # cvi = NIR.subtract(RED).divide(NIR.add(RED).subtract(GREEN)).multiply(0.5).rename('CVI')


    return image.addBands([ndvi, cvi, ndre, gndvi, rvi])


def reduceToMean(image, site, band_name):
    return image.select(band_name).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=site,
        scale=10
    ).get(band_name)


def extract_properties(image, site, vi_list):
    return {vi: reduceToMean(image, site, vi) for vi in vi_list}


def images2features(images, site, vi_lst, img_key):
    if 'sentinel' in img_key:
        return images.map(lambda image: ee.Feature(None, extract_properties(image, site, vi_lst))).getInfo()
    elif 'landsat' in img_key:
        return images.map(lambda image: ee.Feature(None, {
            'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
            **extract_properties(image, site, vi_lst)
        })).getInfo()

def get_take_date(feature, properties, img_key):
    return datetime.strptime(feature['id'][:8], "%Y%m%d").strftime("%Y-%m-%d") if 'sentinel' in img_key else properties['date']
    # if 'sentinel' in img_key:
    #     return datetime.strptime(feature['id'][:8], "%Y%m%d").strftime("%Y-%m-%d")
    # elif 'landsat' in img_key:
    #     return properties['date']

def images2df(images, site, vi_lst, img_key):
    features = images2features(images, site, vi_lst, img_key)
    data = []
    for feature in features['features']:
        properties = feature['properties']
        if all(properties[key] is not None for key in vi_lst):
            take_date = get_take_date(feature, properties, img_key)
            data_lst = [take_date] + [properties[vi] for vi in vi_lst]
            data.append(data_lst)

    columns = ['Date'] + vi_lst
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_satellite_df(start_date, end_date, site, img_key, imgs, vi_lst, cloud_filter):
    if 'sentinel' in img_key:
        cloud_percentage_band = 'CLOUDY_PIXEL_PERCENTAGE'
    elif 'landsat' in img_key:
        cloud_percentage_band = 'CLOUD_COVER'
    filtered_imgs = imgs.filterDate(start_date, end_date).filterBounds(site)

    if cloud_filter:
        processed_img_collection = (filtered_imgs
                                    .map( lambda image: mask_clouds(image, img_key))
                                    .map(lambda image: apply_scale_factors(image, img_key))
                                    .map(lambda image: resampling(image, img_key))
                                    .map(lambda image: add_indices_band(image, img_key)))
    else:
        processed_img_collection = (filtered_imgs
                                    .map(lambda image: apply_scale_factors(image, img_key))
                                    .map(lambda image: resampling(image, img_key))
                                    .map(lambda image: add_indices_band(image, img_key)))

    return images2df(processed_img_collection, site, vi_lst, img_key)


def preprocess_satellite_df(step_dct, df):
    closest_df = pd.DataFrame()
    for step, target_date in step_dct.items():
        target_date = datetime.strptime(target_date, '%Y%m%d')
        closest_row = df.iloc[(df['Date'] - target_date).abs().argsort()[:1]]
        closest_row = closest_row.copy()
        closest_row['Step'] = step
        closest_df = pd.concat([closest_df, closest_row])

    return closest_df


def save_satellite_df(geo_dct, step_dct, images_dct, vi_lst, start_date, end_date, output_filename, cloud_filter=False):
    satellite = pd.DataFrame()
    for img_key, images in images_dct.items():
        for site_name, site in tqdm.tqdm(geo_dct.items(), desc=f'{img_key} processing'):
            df = get_satellite_df(start_date, end_date, site, img_key, images, vi_lst, cloud_filter)
            df = preprocess_satellite_df(step_dct, df)
            df[['Satellite', 'Site']] = img_key, site_name
            satellite = pd.concat([satellite, df])

    satellite['Site'] = satellite['Site'].apply(lambda x: x.split('_')[-1])
    satellite = satellite.groupby(['Site', 'Date', 'Step', 'Satellite']).mean().reset_index()
    satellite.to_csv(output_filename, index=False, encoding='utf-8-sig')

def get_geo_dct(geometry_filename):
    df = pd.read_csv(geometry_filename).drop(columns=['system:index'])
    geo_dct = {row['name']: ee.Geometry.MultiPolygon(dict(json.loads(row['.geo']))['coordinates']) for idx, row in
               df.iterrows()}
    return geo_dct

def main():
    input_dir = '../input'
    output_dir = make_dir('../output')
    save_dir = os.path.join(output_dir, 'eda_process')
    os.makedirs(save_dir, exist_ok=True)


    geometry_filename = os.path.join(input_dir, 'geometry.csv')

    start_date, end_date = '2023-01-01', '2023-06-30'

    step_dct = {'분얼전': '20230130', '분얼전기': '20230321', '분얼후기': '20230417',
                '개화기': '20230501', '개화2주후': '20230520', '개화4주후': '20230601', '수확': '20230615'}

    images_dct = {'sentinel2': (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")),
                  'landsat8': ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
                  'landsat9': ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")}

    vi_lst = ['NDVI', 'CVI', 'NDRE', 'GNDVI', 'RVI']

    geo_dct = get_geo_dct(geometry_filename)

    check_points = ['cloud', 'raw']
    for check_point in check_points:
        output_filename = os.path.join(save_dir, f'satellite_{check_point}.csv')
        cloud_filter = True if check_point == 'cloud' else False
        save_satellite_df(geo_dct, step_dct, images_dct, vi_lst, start_date, end_date, output_filename, cloud_filter)


if __name__ == '__main__':
    main()
