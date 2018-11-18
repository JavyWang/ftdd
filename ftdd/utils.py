# -*- coding: utf-8 -*-

"""
2018-10-14 @Javy Wang
Functions for data processing, graph plot, math operation,
"""
import numpy as np
import os
import pandas as pd
import time
import re
from matplotlib import pyplot as plt
import math
import networkx as nx
import csv


# ----------------------------------------------------------------------
# file read and write functions
# these function are used to write node_list, edge_list, node_attr, edge_attr of the road graph to file and read it again, which is easy to create the network


def dict_to_csv(file_name, data_dict):
    """
    功能：将一字典写入到csv文件中

    :param file_name:
    :param data_dict:
    :return:
    """
    with open(file_name, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in data_dict.iteritems():
            csvWriter.writerow([k, v])
        csvFile.close()


def csv_to_dict(file_name, keyIndex=0, valueIndex=1):
    """
    功能：将一字典写入到csv文件中(尚未写完)

    :param file_name:
    :return:
    """
    dataDict = {}
    with open(file_name, "r") as csvFile:
        dataLine = csvFile.readline().strip("\n")
        while dataLine != "":
            tmpList = dataLine.split(',')
            dataDict[tmpList[keyIndex]] = tmpList[valueIndex]
            dataLine = csvFile.readline().strip("\n")
        csvFile.close()
    return dataDict


# ----------------------------------------------------------------------
# data processing


def file_rename():
    """
    该函数用于对文件重命名
    :return:
    """
    path = input('请输入文件路径(结尾加上/)')
    # 获取该目录下所有文件，存入列表中
    f = os.listdir(path)
    n = 0
    for i in f:
        # 设置旧文件名（就是路径+文件名）
        old_name = path + f[n]
        # 设置新文件名
        new_name = path + 'a' + str(n + 1) + '.JPG'
        # 用os模块中的rename方法对文件改名
        os.rename(old_name, new_name)
        print(old_name, '======>', new_name)
        n += 1


def data_init(data_path, save_path):
    """
    数据初始化包含以下操作：
        1. 数据读取：创建 dataframe 存储数据，命名为 cab_trace_df
        2. 数据标记：将文件名（类似 abboip 等字符）添加到 cab_trace_df 中，并对文件进行编号（从1开始）和排序
        3. 数据存储：存储为csv格式的文件

    :arg:
        data_path: the file path of source data
        save_path: the file path of output data
    :return:
    """
    files = os.listdir(data_path)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        source_file_name = re.findall(r"new_(.+?).txt",
                                      files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = data_path + files[file_id]
        print(source_file)

        # 读入数据
        cab_trace_df = pd.read_csv(source_file, sep=' ',
                                   names=['latitude', 'longitude', 'fare',
                                          'UNIX_time'])
        print(cab_trace_df)

        # 对cab_trace_df进行标记操作，加上出租车编号，文件名，按照时间顺序重排
        cab_trace_df['cab_id'] = file_id + 1  # 出租车编号从1开始
        cab_trace_df['cab_name'] = source_file_name[0]  # 保留出租车名
        cab_trace_df.sort_index(by='UNIX_time', axis=0, ascending=True,
                                inplace=True)  # 原文件是时间倒序, 设置inplace为True 使之生效
        print(cab_trace_df)

        # 存储数据到目标文件夹
        save_file_name = save_path + str(file_id + 1) + '.csv'  # 文件名
        cab_trace_df.to_csv(save_file_name, sep=',', index=False,
                            columns=['cab_id', 'cab_name', 'latitude',
                                     'longitude', 'fare', 'UNIX_time'])

        file_id += 1  # 文件编号自加1

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f' % cost_time)  # 显示用时，保留两位小数


def data_joint(input_path, output_path):
    """
    该函数用于将出租车的单个轨迹点拼接成由相邻轨迹点构成的一段子轨迹
    数据连接包含以下操作：
        1. 导入每个csv文件，存为 cab_trace_df, 用两个子 dataframe实现拼合操作
        2. 存储为csv文件

    :arg:
        input_path: the file path of input data
        output_path: the file path of output data
    :return:
    """
    # 遍历文件夹
    files = os.listdir(input_path)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print(source_file)

        # 读入数据
        cab_trace_df = pd.read_csv(source_file)
        print("输出 cab_trace_df")
        print(cab_trace_df)
        columns_cab_trace_df = cab_trace_df.shape[
            0]  # return the number of columns, shape[1] means the number of rows

        # 拼接数据列名为 'cab_id', 'cab_name', 'latitude1', 'longitude1', 'fare1', 'UNIX_time1'，'latitude2', 'longitude2', 'fare2', 'UNIX_time2'
        cab_trace_df1 = cab_trace_df.drop(
            [columns_cab_trace_df - 1])  # 删除最后一行数据后赋给cab_trace_df1
        cab_trace_df1.rename(
            columns={'latitude': 'latitude1', 'longitude': 'longitude1',
                     'fare': 'fare1', 'UNIX_time': 'UNIX_time1'},
            inplace=True)  # rename the columns
        cab_trace_df1.reset_index(drop=True, inplace=True)  # reset index from 0
        print("输出 cab_trace_df1")
        print(cab_trace_df1)

        cab_trace_df2 = cab_trace_df.drop([0])  # 删除第一行数据后赋给cab_trace_df2
        cab_trace_df2.drop(['cab_id', 'cab_name'], axis=1,
                           inplace=True)  # delete 'cab_id', 'cab_name' from cab_trace_df2
        cab_trace_df2.rename(
            columns={'latitude': 'latitude2', 'longitude': 'longitude2',
                     'fare': 'fare2', 'UNIX_time': 'UNIX_time2'},
            inplace=True)  # rename the columns
        cab_trace_df2.reset_index(drop=True, inplace=True)  # reset index from 0
        print("输出 cab_trace_df2")
        print(cab_trace_df2)

        cab_trace_df_join = pd.concat([cab_trace_df1, cab_trace_df2], axis=1,
                                      join_axes=[cab_trace_df1.index])
        print("输出 cab_trace_df_join")
        print(cab_trace_df_join)

        # 存储数据到目标文件夹
        save_file_name = output_path + str(source_file_name[0]) + '.csv'  # 文件名
        cab_trace_df_join.to_csv(save_file_name, sep=',', index=False,
                                 columns=['cab_id', 'cab_name', 'latitude1',
                                          'longitude1', 'fare1', 'UNIX_time1',
                                          'latitude2', 'longitude2', 'fare2',
                                          'UNIX_time2'])

        file_id += 1  # 文件编号自加1

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def data_stamp(input_path, output_path):
    """
    数据标记包含以下操作：
        1. 标记出载客上下车位置，即一段轨迹的起始点；
        2. 标记出轨迹的序号，即每个出租车的轨迹序列；

    :arg:
        input_path: the file path of input data
        output_path: the file path of output data
    :return:
    """
    # 遍历文件夹
    files = os.listdir(input_path)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print(source_file)

        # 读入数据
        cab_trace_df = pd.read_csv(source_file)
        print("输出 cab_trace_df")

        # 添加一列记录-status
        # 找到cab_trace_df中fare记录0和1交换的记录
        # 0到1为上车点（1），1到0为下车点（2），都为0为空车（0），都为1位载客（3）

        cab_trace_df['status'] = -1  # 出租车状态默认都为 -1
        cab_trace_df.loc[(cab_trace_df['fare1'].isin(['0'])) & (
            cab_trace_df['fare2'].isin(['0'])), 'status'] = '0'
        cab_trace_df.loc[(cab_trace_df['fare1'].isin(['0'])) & (
            cab_trace_df['fare2'].isin(['1'])), 'status'] = '1'
        cab_trace_df.loc[(cab_trace_df['fare1'].isin(['1'])) & (
            cab_trace_df['fare2'].isin(['0'])), 'status'] = '2'
        cab_trace_df.loc[(cab_trace_df['fare1'].isin(['1'])) & (
            cab_trace_df['fare2'].isin(['1'])), 'status'] = '3'

        # 取出cab_trace_df中status列，统计元素个数
        # 根据status状态，确定轨迹的序列值，构造一个列表存储序列
        # 添加一列记录-sequence，将由列表构造的series赋值给 sequence列

        cab_status = cab_trace_df['status']
        print(cab_status)
        num_cab_status = cab_status.count()  # Return number of non-NA/null observations in the Series
        print(num_cab_status)
        list_b = []
        sequence = 0
        for index in range(0, num_cab_status):
            status = int(cab_status[index])
            if status == 1:
                sequence += 1
                list_b.append(sequence)
            elif status == 2:
                list_b.append(sequence)
            elif status == 3:
                list_b.append(sequence)
            elif status == 0:
                list_b.append(0)

        print(list_b)
        cab_sequence = pd.Series(list_b)
        print(cab_sequence)
        cab_trace_df['sequence'] = cab_sequence

        # 存储数据到目标文件夹
        save_file_name = output_path + str(source_file_name[0]) + '.csv'  # 文件名
        cab_trace_df.to_csv(save_file_name, sep=',', index=False,
                            columns=['cab_id', 'cab_name', 'latitude1',
                                     'longitude1', 'fare1', 'UNIX_time1',
                                     'latitude2', 'longitude2', 'fare2',
                                     'UNIX_time2', 'status', 'sequence'])

        file_id += 1  # 文件编号自加1

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def data_concat(input_path, output_path):
    """
    数据标记包含以下操作：
        1. 将所有的出租车轨迹文件合并为一个文件并保存；
        2. 去掉中间不载客的轨迹记录后保存为一个文件；

    :param input_path: the file path of input data
    :param output_path: the file path of output data
    :return:
    """
    # 遍历文件夹
    files = os.listdir(input_path)
    print(files)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    first_cab_file = input_path + files[0]  # take out the first file in the file list
    all_cab_trace_df = pd.read_csv(
        first_cab_file)  # build a dataframe to cover all cab traces
    print("输出 all_cab_trace_df")
    print(all_cab_trace_df)

    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        print(i)
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print(source_file)
        file_id += 1  # 文件编号自加1

        if int(source_file_name[0]) == 1:
            continue

        # 读入数据
        cab_trace_df = pd.read_csv(source_file)
        print("输出 cab_trace_df")
        print(cab_trace_df)

        all_cab_trace_df = pd.concat([all_cab_trace_df, cab_trace_df], ignore_index=True)  #
        print("合并后输出 all_cab_trace_df")
        print(all_cab_trace_df)

    # all_cab_trace_df.sort_index(by='cab_id', axis=0, ascending=True, inplace=True)   # sort all_cab_trace_df from cab_id=0 to cab_id=534
    print("完整版的 all_cab_trace_df")
    print(all_cab_trace_df)
    all_cab_trace_only_fare_df = all_cab_trace_df.loc[
        (all_cab_trace_df['status'].isin(['1', '2', '3']))]
    print("完整版的 all_cab_trace_only_fare_df")
    print(all_cab_trace_only_fare_df)

    # 存储数据到目标文件夹
    save_file_name = output_path + 'all_cab_trace.csv'  # file name
    all_cab_trace_df.to_csv(save_file_name, sep=',', index=False,
                            columns=['cab_id', 'latitude1',
                                     'longitude1', 'fare1', 'UNIX_time1',
                                     'latitude2', 'longitude2', 'fare2',
                                     'UNIX_time2', 'status', 'sequence'])

    save_file_name1 = output_path + 'all_cab_trace_only_fare.csv'  # file name
    all_cab_trace_only_fare_df.to_csv(save_file_name1, sep=',', index=False,
                                      columns=['cab_id',
                                               'latitude1',
                                               'longitude1', 'fare1',
                                               'UNIX_time1',
                                               'latitude2', 'longitude2',
                                               'fare2',
                                               'UNIX_time2', 'status',
                                               'sequence'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def data_describe(input_path):
    """
    A function that generates descriptive statistics of all_cab_trace_df

    :param input_path: the file path of input data
    :return stat_dict: a dictionary of descriptive statistics of all_cab_trace_df
    """
    stat_dict = dict(
        MIN_LATITUDE=0,
        MAX_LATITUDE=0,
        MIN_LONGITUDE=0,
        MAX_LONGITUDE=0,
        MIN_TIME=0,
        MAX_TIME=0,
        STATUS_0=0,
        STATUS_1=0,
        STATUS_2=0,
        STATUS_3=0,
    )

    start_time = time.time()
    all_cab_trace_file = input_path + 'all_cab_trace.csv'  # take out the first file in the file list
    all_cab_trace_df = pd.read_csv(all_cab_trace_file)  # build a dataframe to cover all cab traces
    print("输出 all_cab_trace_df")
    print(all_cab_trace_df)

    latitude1 = all_cab_trace_df['latitude1'].describe()
    print('输出 latitude1 统计结果')
    print(latitude1)
    longitude1 = all_cab_trace_df['longitude1'].describe()
    print('输出 longitude1 统计结果')
    print(longitude1)
    latitude2 = all_cab_trace_df['latitude2'].describe()
    print('输出 latitude2 统计结果')
    print(latitude2)
    longitude2 = all_cab_trace_df['longitude2'].describe()
    print('输出 longitude2 统计结果')
    print(longitude2)
    UNIX_time1 = all_cab_trace_df['UNIX_time1'].describe()
    print('输出 UNIX_time1 统计结果')
    print(UNIX_time1)
    UNIX_time2 = all_cab_trace_df['UNIX_time2'].describe()
    print('输出 UNIX_time2 统计结果')
    print(UNIX_time2)
    status = all_cab_trace_df['status'].value_counts()
    print('输出 status 统计结果')
    print(status)

    stat_dict['MIN_LATITUDE'] = min(latitude1['min'], latitude2['min'])  # set the smaller of both to the minimum
    stat_dict['MAX_LATITUDE'] = max(latitude1['max'], latitude2['max'])  # set the larger of both to the maximum
    stat_dict['MIN_LONGITUDE'] = min(longitude1['min'],
                                     longitude2['min'])  # set the smaller of both to the minimum
    stat_dict['MAX_LONGITUDE'] = max(longitude1['max'],
                                     longitude2['max'])  # set the larger of both to the maximum
    stat_dict['MIN_TIME'] = min(UNIX_time1['min'],
                                UNIX_time2['min'])  # set the smaller of both to the minimum
    stat_dict['MAX_TIME'] = max(UNIX_time1['max'],
                                UNIX_time2['max'])  # set the larger of both to the maximum
    stat_dict['STATUS_0'] = status[0]
    stat_dict['STATUS_1'] = status[1]
    stat_dict['STATUS_2'] = status[2]
    stat_dict['STATUS_3'] = status[3]
    print('输出统计结果 stat_dict')
    print(stat_dict)

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('数据统计用时为：%.2f 分钟' % cost_time)  # 显示用时，保留两位小数

    return stat_dict


def data_processing():
    """
    该函数用于运行数据处理程序
    """
    # Caution！ 这里文件路径要设置为自己数据所在的文件目录
    data_path = {
        'data1': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting1\\',
        # 文件存储目录1
        'data2': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting2\\',
        # 文件存储目录2
        'data3': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting3\\',
        # 文件存储目录3
        'data4': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting4\\',
        # 文件存储目录4
        'data5': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting5\\',
        # 文件存储目录5
    }
    # data_init(data_path['data1'], data_path['data2'])
    # data_joint(data_path['data2'], data_path['data3'])
    # data_stamp(data_path['data3'], data_path['data4'])
    # data_concat(data_path['data4'], data_path['data5'])
    data_describe(data_path['data5'])


def data_filter(input_file, long_base, lati_base, long_term, lati_term):
    """
    The function is used to filter those traces that are over the bound of the grid
    :return:
    """
    trace_df = pd.read_csv(input_file)  # build a dataframe to cover all cab traces
    trace_df.index = range(1, len(trace_df) + 1)  # 行索引从1开始，即节点编号从1开始
    print("输出 trace_df")
    print(trace_df)

    # take out those traces that longitudes and latitudes all are within this range
    sub_trace_df = trace_df[(trace_df['latitude1'] >= lati_base)
                            & (trace_df['latitude1'] <= lati_term)
                            & (trace_df['latitude2'] >= lati_base)
                            & (trace_df['latitude2'] <= lati_term)
                            & (trace_df['longitude1'] >= long_base)
                            & (trace_df['longitude1'] <= long_term)
                            & (trace_df['longitude2'] >= long_base)
                            & (trace_df['longitude2'] <= long_term)
                            ]
    print(sub_trace_df)
    sub_trace_df.to_csv('net_data.csv', sep=',', index=False)
    return sub_trace_df


def mem_usage(pandas_obj):
    """
    return the memory usage of dataframe
    :param pandas_obj:
    :return:
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def data_transfer(input_path, output_path):
    """
    数据转换包含以下操作：
        1. 去掉出租车中的车名一列；
        2. 将经纬度减去基准值后保留四位有效数值，再乘以10000转为整形；

    :param input_path: the file path of input data
    :param output_path: the file path of output data
    :return:
    """
    # 遍历文件夹
    files = os.listdir(input_path)
    print(files)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        print(i)
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print(source_file)
        file_id += 1  # 文件编号自加1

        # 读入数据
        cab_trace_df = pd.read_csv(source_file)
        print("输出 cab_trace_df")
        print(cab_trace_df)

        # delete cab_name
        cab_trace_df.drop(['cab_name'], axis=1, inplace=True)

        # data types transfer
        print("——————转换前数据类型——————")
        print(mem_usage(cab_trace_df))
        print("————转换前latitude1的类型————")
        print(cab_trace_df['latitude1'].dtype)

        f_latitude = lambda x: (x - 32) * 10000
        f_longitude = lambda x: (0 - x - 115) * 10000
        cab_trace_df['latitude1'] = cab_trace_df['latitude1'].map(f_latitude)
        cab_trace_df['longitude1'] = cab_trace_df['longitude1'].map(f_longitude)
        cab_trace_df['latitude2'] = cab_trace_df['latitude2'].map(f_latitude)
        cab_trace_df['longitude2'] = cab_trace_df['longitude2'].map(f_longitude)

        cab_trace_df[['latitude1', 'longitude1', 'latitude2', 'longitude2']] = cab_trace_df[
            ['latitude1', 'longitude1', 'latitude2', 'longitude2']].astype(int)
        cab_trace_df = cab_trace_df.apply(pd.to_numeric, downcast='signed')  # errors='ignore'
        print("——————转换后的数据——————")
        print(cab_trace_df)
        print("——————转换后数据类型——————")
        print(mem_usage(cab_trace_df))
        print("————转换后latitude1的类型————")
        print(cab_trace_df['latitude1'].dtype)

        # 存储数据到目标文件夹
        save_file_name = output_path + str(source_file_name[0]) + '.csv'  # 文件名
        cab_trace_df.to_csv(save_file_name, sep=',', index=False,
                            columns=['cab_id', 'latitude1',
                                     'longitude1', 'fare1', 'UNIX_time1',
                                     'latitude2', 'longitude2', 'fare2',
                                     'UNIX_time2', 'status', 'sequence'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def data_detection(input_path, output_path):
    """
    find the records that latitude and longitude value are out of given range in each cab file

    :param input_path:
    :param output_path:
    :return:
    """
    # 遍历文件夹
    files = os.listdir(input_path)
    print(files)
    count_of_files = len(files)
    print(count_of_files)

    start_time = time.time()
    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        print(i)
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出出租车名
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print("output the file name")
        print(source_file)
        file_id += 1  # 文件编号自加1

        # 读入数据
        cab_trace_df = pd.read_csv(source_file)
        # print("————source file————")
        # print("输出 cab_trace_df")
        # print(cab_trace_df)

        # find these records within given range
        # cab_trace_df = cab_trace_df[(cab_trace_df['latitude1'] >= 52000) & (cab_trace_df['latitude1'] <= 60000)]
        # cab_trace_df = cab_trace_df[(cab_trace_df['longitude1'] >= 70000) & (cab_trace_df['longitude1'] <= 76000)]

        # find these records out of given range
        # cab_trace_df = cab_trace_df[(cab_trace_df['latitude1'] < 52000) | (cab_trace_df['latitude1'] > 60000) |
        #                             (cab_trace_df['longitude1'] < 70000) | (cab_trace_df['longitude1'] > 76000)]
        # print("————the records out of given range————")
        # print(cab_trace_df)

        f_latitude = lambda x: (x - 50000)
        f_longitude = lambda x: (x - 70000)
        cab_trace_df['latitude1'] = cab_trace_df['latitude1'].map(f_latitude)
        cab_trace_df['longitude1'] = cab_trace_df['longitude1'].map(f_longitude)
        cab_trace_df['latitude2'] = cab_trace_df['latitude2'].map(f_latitude)
        cab_trace_df['longitude2'] = cab_trace_df['longitude2'].map(f_longitude)

        cab_trace_df[['latitude1', 'longitude1', 'latitude2', 'longitude2']] = cab_trace_df[
            ['latitude1', 'longitude1', 'latitude2', 'longitude2']].astype(int)
        cab_trace_df = cab_trace_df.apply(pd.to_numeric, downcast='signed')   # errors='ignore'

        if cab_trace_df.empty is True:
            print(" %d cab is in the range! " % file_id)
        else:
            # 存储数据到目标文件夹
            save_file_name = output_path + str(source_file_name[0]) + '.csv'  # 文件名
            cab_trace_df.to_csv(save_file_name, sep=',', index=False,
                                columns=['cab_id', 'latitude1',
                                         'longitude1', 'fare1', 'UNIX_time1',
                                         'latitude2', 'longitude2', 'fare2',
                                         'UNIX_time2', 'status', 'sequence'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def data_handling(input_path, output_path):
    """
    # 先构造一个dataframe，包含以下列['gird_id', 'pick_up', 'drop_off', weight]

    :param input_path:
    :param output_path:
    :return:
    """

    start_time = time.time()
    # load data
    cab_file = input_path + 'all_cab_trace_only_fare.csv'  # 文件名
    cab_df = pd.read_csv(cab_file)  # all trajectories with fare
    cab_df[['latitude1', 'latitude2', 'longitude1', 'longitude2']] = cab_df[
        ['latitude1', 'latitude2', 'longitude1', 'longitude2']].astype(int)
    cab_df = cab_df.apply(pd.to_numeric, downcast='signed')
    cab_df1 = cab_df[['latitude2', 'longitude2', 'status']][cab_df['status'].isin([1])]  # take out the df that status = 1, pick up site
    cab_df2 = cab_df[['latitude1', 'longitude1', 'status']][cab_df['status'].isin([2])]  # take out the df that status = 2, drop off site
    cab_df1.rename(columns={'latitude2': 'latitude', 'longitude2': 'longitude'}, inplace=True)
    cab_df2.rename(columns={'latitude1': 'latitude', 'longitude1': 'longitude'}, inplace=True)
    print(cab_df1.head())
    print(cab_df2.head())
    # latitude:2000-10000   区间长度：8000   一格160
    # longitude:0-6000      区间长度：6000   一格200   , so total there is 30*50=1500 grids
    unit_latitude = 160
    unit_longitude = 200
    f_grid_id = lambda x: int((x.latitude - 2000) / unit_latitude) * 30 + int((x.longitude / unit_longitude)) + 1
    cab_df1['grid_id'] = cab_df1.apply(f_grid_id, axis=1)
    cab_df2['grid_id'] = cab_df2.apply(f_grid_id, axis=1)
    cab_df1.to_csv('grid01.csv', sep=',', index=False, columns=['latitude', 'longitude', 'status', 'grid_id'])
    cab_df2.to_csv('grid02.csv', sep=',', index=False, columns=['latitude', 'longitude', 'status', 'grid_id'])
    # cab1_file = input_path + 'grid01.csv'  # 文件名
    # cab2_file = input_path + 'grid02.csv'  # 文件名
    # cab_df1 = pd.read_csv(cab1_file)
    # cab_df2 = pd.read_csv(cab2_file)
    # cab_df1 = cab_df1.apply(pd.to_numeric, downcast='signed')
    # cab_df2 = cab_df2.apply(pd.to_numeric, downcast='signed')
    print(cab_df1.head())
    print(cab_df2.head())
    print(cab_df1.dtypes)
    print(cab_df2.dtypes)
    # concat the two cab_df
    cab_df = pd.concat([cab_df1, cab_df2], ignore_index=True)
    cab_df = cab_df[cab_df['grid_id'] > 0]  # there exists value under zero
    grid_df = pd.DataFrame(columns=('grid_id', 'pick_up', 'drop_off', 'weight'))
    cab_df_gp = cab_df.groupby(['grid_id'])
    for group in cab_df_gp:
        grid = group[1]
        grid_id = int(grid.head(1)['grid_id'])
        pick_up = grid['status'][grid['status'].isin([1])].count()
        drop_off = grid['status'][grid['status'].isin([2])].count()
        weight = pick_up + drop_off
        grid_df = grid_df.append({'grid_id': grid_id, 'pick_up': pick_up, 'drop_off': drop_off, 'weight': weight}, ignore_index=True)

    save_file_name = output_path + 'grid.csv'  # 文件名
    grid_df.to_csv(save_file_name, sep=',', index=False, columns=['grid_id', 'pick_up', 'drop_off', 'weight'])
    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数
# ----------------------------------------------------------------------
# network functions


# ----------------------------------------------------------------------
# graph functions


def plot_graph(road_graph):
    """
    The function is used to plot the road graph
    :return:
    """
    nx.draw(road_graph, with_labels=True)
    nx.spring_layout(road_graph)
    plt.show()
    print('This is the figure of this road graph.')


def plot_grid_density(road_graph):
    """
    The function is used to plot the density distribution in the grid
    :param road_graph:
    :return:
    """


# ----------------------------------------------------------------------
# math functions


def my_activation_function(base):
    """
    该函数的作用是绘制出所构造的激活函数的图像

    :arg:
        base: 基数，即对数函数的底数, 0 < base < 1
    :return:
        无返回值
    """
    plt.figure()
    plt.subplot(111)
    x = np.arange(0.001, 1.0, 0.001)
    y = [1 / (1 + math.log(a, base)) for a in x]
    plt.plot(x, y, linewidth=2, color="#007500", label='1/(1+log' + str(base) + '(x))')
    plt.plot([1, 1], [y[0], y[-1]], "r--", linewidth=2)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
