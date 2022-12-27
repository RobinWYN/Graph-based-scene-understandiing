#风险评估软件开发  第一版
#最后一次更新 2022年7月11日

from matplotlib import pyplot as plt
from matplotlib import axes
from matplotlib import cm
import pandas as pa
import os
import json
import numpy as np
from importlib import import_module
from common.protocal.Field_protocal import *
from common.protocal.object_protocal import *
from common.protocal.config_protocal import *
from config_reader.config import *
from visualization.visualization import *
import matplotlib.pyplot as pyplot

class SafetyField:
    def __init__(self) -> None:
        pass

    # 安全场生成函数
    def generate(self, data,config):
        safety_field = Field(config.Resolution)
        safety_field.transmat()
        # pyplot.imshow(safety_field.field)
        # pyplot.show()

        # 将数据送入处理算法
        for items in data[1]:
            [subField, pos] = config.label_to_algorithm[items.label].generate(items, config.Resolution, config.RealSize)
            pos = [np.int(items.y/config.RealSize) + pos[0], np.int(items.x/config.RealSize) + pos[1]]
            # pos = [np.int(items.x/config.RealSize) + pos[0], np.int(items.y/config.RealSize) + pos[1]]
            subField.transmat()
            config.adder.add(safety_field, subField, pos)
            # config.reprocessor.run(safety_field)
            # pyplot.imshow(safety_field.field)
            # pyplot.show()

        # 输出场
        # print(np.max(safety_field.field))
        safety_field.transmat()
        return safety_field

# main函数是一个使用demo，真正使用的时候只需要按照类似的方法依次调用函数即可
if __name__=='__main__':

    my_safetyf = SafetyField()
    
    # 第一步,读取配置
    my_config = config_reader().read()
    # 第二步，从适配器拿数据
    for i in range(3):
        
        # raw_data = []
        # my_config.adaptor.dataInput(raw_data)
        data = my_config.adaptor.get_data()
        # 第三步，调用安全场生成函数（在完整系统中使用时，这一步返回的场就可以供下一步使用了）
        safety_field = my_safetyf.generate(data,my_config)
        # 第四步，后处理
        my_config.reprocessor.run(safety_field)
        print(np.max(safety_field.field))
        # 第五步，可视化
        visualization().display(safety_field.field, data, my_config.Resolution, my_config.RealSize)



