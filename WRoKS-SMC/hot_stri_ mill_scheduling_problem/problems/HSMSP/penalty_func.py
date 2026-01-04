import pandas as pd
import math

def read_in(path):
    rename_dict = {
        '宽度惩罚点数': 'width_Punishment',
        '厚度反跳惩罚': 'thick_-',
        '厚度正跳惩罚': 'thick_+',
        '硬度跳跃惩罚': 'hard_Punishment',
        '出炉温度跳跃惩罚': 'oven_temperature',
        '卷取温度跳跃惩罚': 'crimp_temperature',
        '精轧温度惩罚': 'finishing_temperature'
    }

    try:
        df = pd.read_excel(path, engine='xlrd')
        df.rename(columns=rename_dict, inplace=True)

        # print("Columns after renaming:")
        # print(df.columns)

        return df

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

filename = r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\penalty_func.xls'

data = read_in(filename)


def get_p_width(delta):
    delta = int(delta)
    if 0 <= delta <= 359:
        return data['width_Punishment'].get(delta, 1000)
    else:
        return 1000

def get_p_hard(hard):
    hard = abs(hard)
    return data['hard_Punishment'].get(hard, 1000)

def get_p_thick(thick):
    thick_int = math.ceil(thick)
    if thick == 0:
        return 0

    elif thick > 0:
        thickness_penalty = data['thick_+'].get(thick_int, 1000) * (thick / thick_int)
    else:
        thickness_penalty = data['thick_-'].get(abs(thick_int), 1000) * (thick_int / thick)

    return thickness_penalty
