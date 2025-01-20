import pandas as pd
import numpy as np

# 读取数据
building_dni_df = pd.read_csv(r"D:\Desktop\Data\buildings_DNI.csv")
building_sis_df = pd.read_csv(r"D:\Desktop\Data\buildings_SIS.csv")

# 确保两个数据集能通过 "OID_" 列匹配
assert all(building_dni_df['OID_'] == building_sis_df['OID_'])

# 创建一个新的 DataFrame 来存储结果
result_df = building_dni_df[['OID_', 'use', 'age', 'relhmax', 'STORY', 'Shape_Area', 'GFA', 'Shape_Length', 'BuildingPerimeterShapeFactor', 'BuildingVolumeShapeFactor', 'Volume', 'Slope', 'BO']].copy()
total_radiation_dict = {} # 创建一个字典来存储所有的 total_radiation 数据

# 计算屋顶面积
slope_rad = np.radians(building_dni_df['Slope'])
roof_area = building_dni_df['Shape_Area'] / np.cos(slope_rad)
solar_hour_angle = 0 # 太阳时角为正午，即 h = 0

# 遍历每一天进行计算
for date in building_dni_df.columns[27:]: # 
    latitude = np.radians(building_dni_df['Lat'])  # 地理纬度
    n = int(date[4:8]) - 1  # 一年中的第几天，日期格式为 YYYYMMDD
    declination = np.radians(23.44 * np.cos(np.radians((360 * (284 + n)) / 365)))  # 太阳赤纬角
    azimuth_angle = np.radians(building_dni_df['BO'])  # 屋顶方位角
    solar_alititude_angle = np.radians(90 - latitude + declination) # 太阳高度角
    
    # 入射角公式
    cos_theta = np.sin(latitude) * np.sin(declination) * np.cos(slope_rad) \
                - np.cos(latitude) * np.sin(declination) * np.sin(slope_rad) * np.cos(azimuth_angle) \
                + np.cos(latitude) * np.cos(declination) * np.cos(solar_hour_angle) * np.cos(slope_rad) \
                + np.sin(latitude) * np.cos(declination) * np.cos(solar_hour_angle) * np.sin(slope_rad) * np.cos(azimuth_angle) \
                + np.cos(declination) * np.sin(solar_hour_angle) * np.sin(slope_rad) * np.sin(azimuth_angle)
    
    # 计算直接辐射
    dni_today = building_dni_df[date]
    adjusted_dni = dni_today * cos_theta
    direct_radiation = adjusted_dni * roof_area
    
    # 计算散射辐射
    sis_today = building_sis_df[date]
    diffuse_radiation = sis_today - dni_today * np.cos(solar_alititude_angle)
    diffuse_radiation_on_roof = diffuse_radiation * roof_area * (1 + np.cos(slope_rad)) / 2
    
    # 计算总辐射
    total_radiation = direct_radiation + diffuse_radiation_on_roof # 单位为 Wh
    
    # 保留两位小数,将结果存储在字典中
    total_radiation_rounded = total_radiation.round(2)
    total_radiation_dict[date] = total_radiation_rounded

# 将字典转换为 DataFrame
total_radiation_df = pd.DataFrame(total_radiation_dict)

# 转换为更高效的数据类型
total_radiation_df = total_radiation_df.astype(np.float32)

# 将原始 DataFrame 和包含 total_radiation 数据的新 DataFrame 进行合并
result_df = pd.concat([result_df, total_radiation_df], axis=1)

# 将结果保存为新的 CSV 文件
result_df.to_csv(r'D:\Desktop\Data\buildings_Roof_Solar-2.csv', index=False)