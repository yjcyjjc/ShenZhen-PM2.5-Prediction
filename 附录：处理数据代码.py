import pandas as pd
air_quality_data = pd.read_csv('../data/air_quality_data.csv')
weather_data = pd.read_csv('../data/weather_data.csv')
pm_detail=pd.read_csv('../data/pm_detail_data.csv')#加载数据
# 温度 (0.1°C → °C)
weather_data["T"] = weather_data["T"] / 10
# 降水 (0.1mm → mm)
weather_data["R"] = weather_data["R"] / 10
# 风速 (0.1m/s → m/s)
weather_data["FS"] = weather_data["FS"] / 10
# 气压 (0.1hPa → hPa)
weather_data["P"] = weather_data["P"] / 10
weather_column_mapping = {
    "DDATETIME": "datetime",         #日期时间
    "T": "temperature_C",            #温度(°C)
    "U": "relative_humidity_pct",    #相对湿度(%)
    "R": "precipitation_mm",         #降水(mm)               
    "FS": "wind_speed_ms",           #风速( m/s)
    "FX": "wind_direction_deg",      #风向(°)
    "P": "pressure_hPa",             #气压(hPa)
    "V": "visibility_m"              #能见度(m)
}
weather_data = weather_data.rename(columns=weather_column_mapping)
weather_data["datetime"] = pd.to_datetime(weather_data["datetime"])
air_quality_data_columns_mapping = {
    "JCSJ": "datetime",          #监测时间
    "CDMC": "district",           #监测站(区)
}
air_quality_data = air_quality_data.rename(columns=air_quality_data_columns_mapping)
weather_data["datetime"] = pd.to_datetime(
    weather_data["datetime"],
    format="%Y-%m-%d %H:%M:%S.%f"
)
pm_detail_column_mapping = {
    "JCSJ": "datetime",            #监测时间
    "JCDWMC": "site_id",           #监测站
    "HOUR": "average_pm25_hour"    #小时平均PM2.5浓度(μg/m³)
}
pm_detail = pm_detail.rename(columns=pm_detail_column_mapping)
weather_data.drop(['CRTTIME','DDATETIME_lag1', 'delta', 'delta_min', 'ord'], axis=1, inplace=True)
weather_data.dropna(inplace=True)
air_quality_data.drop(['SYWRW','KQDJ','AQI','ID', 'UPDATESTATUS','CBWRW', 'CDBM','UPDATETIME',
       'CD_BATCH'], axis=1, inplace=True)
air_quality_data.dropna(inplace=True)
air_quality_data['datetime'] = pd.to_datetime(air_quality_data['datetime'], format='mixed')
pm_detail=pm_detail[["datetime", "site_id", "average_pm25_hour"]]
pm_detail.dropna(inplace=True)
particle_to_district = {

    # 南山区
    "华侨城": "南山区",
    "南油": "南山区",
    "前海": "南山区",
    "南海": "南山区",

    # 福田区
    "莲花山": "福田区",
    "莲花": "福田区",
    "荔园": "福田区",
    "通心岭": "福田区",

    # 罗湖区
    "南湖": "罗湖区",
    "洪湖": "罗湖区",

    # 盐田区
    "盐田": "盐田区",
    "梅沙": "盐田区",

    # 龙岗区
    "横岗": "龙岗区",
    "龙岗": "龙岗区",
    "龙城": "龙岗区",
    "黄阁路": "龙岗区",

    # 宝安区
    "西乡": "宝安区",
    "福永": "宝安区",
    "松岗": "宝安区",
    "沙井": "宝安区",

    # 光明区
    "光明": "光明区",
    "公明": "光明区",
    "下陂": "光明区",

    # 龙华区
    "民治": "龙华区",
    "观澜": "龙华区",

    # 坪山区
    "坪山": "坪山区",

    # 大鹏新区
    "葵涌": "大鹏新区",
    "南澳": "大鹏新区",
}#通过查阅站点和区的关系，将站点映射到对应的区域
pm_detail["district"] = pm_detail["site_id"].map(particle_to_district)
pm_detail["datetime"] = pd.to_datetime(
    pm_detail["datetime"],
    format="mixed",
    errors="raise"
)
start_time = pd.to_datetime("2020-01-01 00:00:00")
end_time = pd.to_datetime("2025-12-31 23:59:59")
weather_data = weather_data[
    (weather_data["datetime"] >= start_time) &
    (weather_data["datetime"] <= end_time)
]
air_quality_data = air_quality_data[
    (air_quality_data["datetime"] >= start_time) &
    (air_quality_data["datetime"] <= end_time)
]
pm_detail = pm_detail[
    (pm_detail["datetime"] >= start_time) &
    (pm_detail["datetime"] <= end_time)
]
weather_data.sort_values(by='datetime', inplace=True)
air_quality_data.sort_values(by=['datetime','district'], inplace=True)
pm_detail.sort_values(by=['datetime','district','site_id'], inplace=True)
weather_data.to_csv('../data/processed_weather_data.csv', index=False)
air_quality_data.to_csv('../data/processed_air_quality_data.csv', index=False)
pm_detail.to_csv('../data/processed_pm_detail.csv', index=False)