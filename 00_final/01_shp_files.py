import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Polygon
import warnings
import tqdm

warnings.simplefilter("ignore", UserWarning)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def code_info(input_dir, output_dir, d_dir):
    code_filename = os.path.join(output_dir, '도엽코드_파일경로.csv')
    if not os.path.exists(code_filename):

        info = pd.read_csv(os.path.join(input_dir, '국토교통부 국토지리정보원_수치지도_주소도엽매칭_20231107.csv'), encoding='cp949')
        sido = pd.read_csv(os.path.join(input_dir, '국토교통부 국토지리정보원_구지형도_시도코드_20231107.csv'), encoding='cp949')

        file_data = []

        for root, dirs, files in os.walk(d_dir):
            for file in files:
                if file.endswith('.shp'):
                    shp_file_path = os.path.join(root, file)
                    lnx = file.split('_')[1].replace('.shp', '')


                    file_data.append((shp_file_path, lnx))

        df = pd.DataFrame(file_data, columns=['path', 'lnx'])
        df['lnx'] = df['lnx'].astype(str)
        info['시도코드'] = info['시군구코드'].apply(lambda x: int(str(x)[:2]))
        info = pd.merge(sido, info, on='시도코드', how='inner')
        info = pd.merge(info, df, left_on='도엽코드', right_on='lnx', how='inner')
        info = info.drop(columns=['축척', '중간X값', '중간Y값', 'lnx', '지도종류', '비고', '시도코드'])

        info.to_csv(code_filename, encoding='utf-8-sig', index=False)
    else:
        info = pd.read_csv(code_filename)
    return info

def polygon_to_list(polygon):
    if isinstance(polygon, Polygon):
        return [[[x, y] for x, y in polygon.exterior.coords]]
    else:
        raise ValueError("The geometry is not a Polygon")


def merge_shp(info, sg, d_dir):
    sg_info = info[info['시군구'] == sg]
    sg_path = list(sg_info['path'].unique())

    lst_gdf = []
    for path in sg_path:
        gdf = gpd.read_file(path)
        lst_gdf.append(gdf)

    merged_gdf = gpd.GeoDataFrame(pd.concat(lst_gdf, ignore_index=True), crs=lst_gdf[0].crs)

    output_folder = os.path.join(d_dir, sg)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{sg}.shp')
    merged_gdf.to_file(output_path, encoding='utf-8')


def main():
    input_dir = make_dir('./input')
    output_dir = make_dir('./output')
    d_dir = r'D:\Projects\202406_토지피복도병합'

    info = code_info(input_dir, output_dir, d_dir)
    lst_sg = ['김제시', '남해군', '부안군']

    for sg in tqdm.tqdm(lst_sg, total=len(lst_sg)):
        merge_shp(info, sg, d_dir)


if __name__ == '__main__':
    main()