import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, tan
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.spatial import KDTree

def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    Vectorized version of the Haversine formula to calculate distances
    between multiple pairs of coordinates.
    """
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius in meters
    R = 6371000  
    return R * c

def can_cast_shadow_vectorized(height_i, height_j, distance, sun_altitude):
    """
    Vectorized version to check if a building can cast a shadow on another.
    """
    height_diff = height_i - height_j
    shadow_length = height_diff / np.tan(sun_altitude)
    return shadow_length >= distance

def process_buildings(i, coords, sun_altitude, query_radius, kdtree, representative_hours):
    shadow_counter = np.zeros(len(coords))  # 初始化阴影计数器
    indices = kdtree.query_ball_point(coords[i, 1:3], query_radius, p=2)

    for hour in representative_hours:
        for j in indices:
            if i >= j:
                continue
            distance = haversine_vectorized(*coords[i, 1:3], *coords[j, 1:3])
            if distance < query_radius and can_cast_shadow_vectorized(coords[i, 3], coords[j, 3], distance, sun_altitude[i]):
                shadow_counter[j] += 1

    return shadow_counter

def process_season(season, declination, coords, kdtree, query_radius):
    representative_days = np.arange(0, 91, 7)  # 每个季节间隔7天
    sunrise_hours = 6   # 假设日出时间为早上6点
    sunset_hours = 18   # 假设日落时间为晚上6点
    representative_hours = np.arange(sunrise_hours, sunset_hours + 1, 2)  # 从日出开始，每隔2小时

    total_hours = len(representative_days) * len(representative_hours)
    sun_altitude = np.radians(90) - np.radians(coords[:, 3]) + declination

    total_shadow_counter = np.zeros(len(coords))  # 初始化整个季节的阴影计数器
    with ThreadPoolExecutor(max_workers=10) as executor:
        for day in representative_days:
            futures = []
            for i in range(len(coords)):
                futures.append(executor.submit(process_buildings, i, coords, sun_altitude, query_radius, kdtree, representative_hours))

            for future in tqdm(futures, desc=f"Processing Buildings for {season} Day {day}", unit="building"):
                total_shadow_counter += future.result()

    significant_shadow = np.where(total_shadow_counter / total_hours > 0.05)
    adjacency_list = [[coords[i, 0], coords[j, 0]] for i in range(len(coords)) for j in significant_shadow[0] if i < j]
    adjacency_df = pd.DataFrame(adjacency_list, columns=['shadowing_building', 'shadowed_building'])
    adjacency_df.to_csv(fr'D:\Desktop\Data\adjacency_list_season\buildings_adjacency_list_{season}_new.csv', index=False)
    return adjacency_df.shape



# 数据加载和初始化
building_dni_df = pd.read_csv(r"D:\Desktop\Data\Buildings_Lon_Lat.csv")
building_dni_df = building_dni_df.sort_values(by='Lat')
coords = building_dni_df[['OID_', 'Lon', 'Lat', 'relhmax']].to_numpy()

# 计算查询半径
lat_increment = 1000 / 111000  # 纬度每增加1度，距离增加111公里
mean_lat = np.radians(building_dni_df['Lat'].mean())
lon_increment = 1000 / (111000 * np.cos(mean_lat))  # 经度每增加1度，距离增加111公里 * cos(纬度)
query_radius = (lat_increment + lon_increment) / 2  # 查询半径为纬度和经度的平均值

# 创建KDTree
kdtree = KDTree(coords[:, 1:3])

# 四季的赤纬角
seasons = {
    'spring': np.radians(23.44 * np.cos(np.radians((360 * (284 + (79 - 1))) / 365))),  # 春分
    'summer': np.radians(23.44),  # 夏至
    'autumn': np.radians(23.44 * np.cos(np.radians((360 * (284 + (262 - 1))) / 365))),  # 秋分
    'winter': np.radians(-23.44)  # 冬至
}

# 处理每个季节的数据
for season, declination in seasons.items():
    shape = process_season(season, declination, coords, kdtree, query_radius)
    print(f"{season.capitalize()} Season: {shape}")


# Example code for KD-Tree optimization
from scipy.spatial import KDTree
# KD-Tree for buildings
kdtree = KDTree(building_coordinates) 
# Loop through each building
for building_i in buildings: 
    # Find nearby buildings within a certain radius 
    nearby_buildings = kdtree.query_ball_point(building_i, query_radius) 
    # Loop through each nearby building
    for building_j in nearby_buildings: 
        # Calculate the distance between building_i and building_j
        distance = haversine(building_i, building_j) 
        # Check if building_i can cast a shadow on building_j
        if shadow_cast(building_i, building_j, distance):
            # Add building_j to the adjacency list of building_i
            add_to_adjacency_list(building_i, building_j) 