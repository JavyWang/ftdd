import os
import pandas as pd
import time
import re
import math



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
    # longitude:0-6000      区间长度：6000   一格200   , so total there is 60*100=6000 grids
    unit_latitude = 80
    unit_longitude = 100
    f_grid_id = lambda x: int((x.latitude - 2000) / unit_latitude) * 60 + int((x.longitude / unit_longitude)) + 1
    f_grid_id_s = lambda x: int((x.latitude1 - 2000) / unit_latitude) * 60 + int((x.longitude1 / unit_longitude)) + 1
    f_grid_id_e = lambda x: int((x.latitude2 - 2000) / unit_latitude) * 60 + int((x.longitude2 / unit_longitude)) + 1
    cab_df['s_node'] = cab_df.apply(f_grid_id_s, axis=1)
    cab_df['e_node'] = cab_df.apply(f_grid_id_e, axis=1)
    cab_df1['grid_id'] = cab_df1.apply(f_grid_id, axis=1)
    cab_df2['grid_id'] = cab_df2.apply(f_grid_id, axis=1)
    cab_df1.to_csv('grid01.csv', sep=',', index=False, columns=['latitude', 'longitude', 'status', 'grid_id'])
    cab_df2.to_csv('grid02.csv', sep=',', index=False, columns=['latitude', 'longitude', 'status', 'grid_id'])
    save_file_name1 = output_path + 'new_cab.csv'  # file name
    cab_df.to_csv(save_file_name1, sep=',', index=False,
                                      columns=['cab_id',
                                               'latitude1',
                                               'longitude1', 'fare1',
                                               'UNIX_time1',
                                               'latitude2', 'longitude2',
                                               'fare2',
                                               'UNIX_time2', 'status',
                                               'sequence','s_node','e_node'])
    print(cab_df.head(10))
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


def add_index(input_path, output_path):
    # 给原来的new_cab添加了index
    start_time = time.time()
    cab_file = input_path + 'new_cab.csv'
    cab_df = pd.read_csv(cab_file)
    cab_status = cab_df['status']
    print(cab_status.head(10))
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
    traj_index = pd.Series(list_b)
    print(traj_index.head())
    cab_df['index'] = traj_index
    save_file_name1 = output_path + 'new_cab1.csv'  # file name
    cab_df.to_csv(save_file_name1, sep=',', index=False,
                  columns=['cab_id',
                           'latitude1',
                           'longitude1', 'fare1',
                           'UNIX_time1',
                           'latitude2', 'longitude2',
                           'fare2',
                           'UNIX_time2', 'status',
                           'sequence', 's_node', 'e_node','index'])
    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('添加轨迹索引用时')
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def get_cab_simple(input_path, output_path):
    # 得到简化版的轨迹数据，只保存司机编号/司机轨迹序号/起点/原点/轨迹序号
    start_time = time.time()
    traj_file = input_path + 'new_cab1.csv'  # 文件名
    cab_trace_df = pd.read_csv(traj_file)  # all trajectories with fare
    traj_df = pd.DataFrame(columns=('cab_id', 'sequence', 's_node', 'e_node','index'))# return inte_sites
    cab_trace_df = cab_trace_df.apply(pd.to_numeric, downcast='signed', errors='ignore')

    # find the num of trajectories in cab_trace_df
    cab_trace_group = cab_trace_df.groupby(['cab_id', 'sequence'])
    for group in cab_trace_group:
        # take out the slice of the traj_id trajectory
        # print(group)
        per_traj_df = group[1]
        s_node_df = per_traj_df.loc[(per_traj_df['status'].isin(['1']))]
        e_node_df = per_traj_df.loc[(per_traj_df['status'].isin(['2']))]
        if s_node_df.empty is False and e_node_df.empty is False:

            s_node = int(s_node_df.head(1)['e_node'])
            e_node = int(e_node_df.head(1)['s_node'])
            cab_id = int(s_node_df.head(1)['cab_id'])
            traj_id = int(s_node_df.head(1)['sequence'])
            index = int(s_node_df.head(1)['index'])
            traj_df = traj_df.append({'cab_id': cab_id, 'sequence': traj_id, 's_node': s_node, 'e_node': e_node, 'index': index}, ignore_index=True)
            print(traj_df.tail())
    save_file_name = output_path + 'new_cab_simple.csv'  # 文件名
    traj_df.to_csv(save_file_name, sep=',', index=False,
                   columns=['cab_id', 'sequence', 's_node', 'e_node','index'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('得到简化版轨迹用时')
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def get_traj(input_path, output_path, output_path1):
    # 得到每对起点和终点之间的超过100条轨迹的数据，分为单个文件和整体文件
    start_time = time.time()
    route_file=input_path+'newgrid.csv'
    route_df = pd.read_csv(route_file)
    cab_simple_file = input_path + 'new_cab_simple.csv'
    cab_simple_df = pd.read_csv(cab_simple_file)
    cab_file = input_path + 'new_cab1.csv'
    cab_df = pd.read_csv(cab_file)
    lens=route_df['grid_id'].count()
    cab_all = pd.DataFrame(columns=['cab_id',
                                    'latitude1',
                                    'longitude1', 'fare1',
                                    'UNIX_time1',
                                    'latitude2', 'longitude2',
                                    'fare2',
                                    'UNIX_time2', 'status',
                                    'sequence', 's_node', 'e_node', 'index'])
    for i in range(lens):
        for j in range(lens):
            if (i!=j) :
                start_node=int(route_df.loc[i, 'grid_id'])
                end_node=int(route_df.loc[j,'grid_id'])
                cab_index = cab_simple_df[(cab_simple_df['s_node'].isin([start_node])) & (
                    cab_simple_df['e_node'].isin([end_node]))]
                num = cab_index['s_node'].count()
                if num >= 100:
                    print("one more S-E nodes")
                    index=list(cab_index['index'])
                    filename=str(start_node)+'-'+str(end_node)

                    cab = cab_df[cab_df['index'].isin(index)]
                    save_file_name1 = output_path + filename+'.csv'  # file name
                    save_file_name2 = output_path1 + filename +'-index' +'.csv' # file name
                    cab_all=pd.concat([cab_all,cab])
                    cab.to_csv(save_file_name1, sep=',', index=False,
                                   columns=['cab_id',
                                            'latitude1',
                                            'longitude1', 'fare1',
                                            'UNIX_time1',
                                            'latitude2', 'longitude2',
                                            'fare2',
                                            'UNIX_time2', 'status',
                                            'sequence', 's_node', 'e_node', 'index'])
                    cab_index.to_csv(save_file_name2,sep=',', index=False,
                   columns=['cab_id', 'sequence', 's_node', 'e_node','index'])

    save_file_name = output_path1 +'all_chosed.csv'
    cab_all.to_csv(save_file_name, sep=',', index=False,
                                   columns=['cab_id',
                                            'latitude1',
                                            'longitude1', 'fare1',
                                            'UNIX_time1',
                                            'latitude2', 'longitude2',
                                            'fare2',
                                            'UNIX_time2', 'status',
                                            'sequence', 's_node', 'e_node', 'index'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('得到符合要求的轨迹的用时')
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def calculate_dist(one_traj):
    # calculate the distance between Adjacent two points
    lat_dist = abs(one_traj['latitude1'] - one_traj['latitude2'])*11.12
    lon_dist = abs(one_traj['longitude1'] - one_traj['longitude2'])*8.76
    dist = int(math.sqrt(pow(lat_dist, 2) + pow(lon_dist, 2)))
    return dist


def calculate(input_path, output_path):
    start_time = time.time()
    filename=input_path+'all_chosed.csv'
    cab=pd.read_csv(filename)
    cab['time'] = 0
    cab['dist'] = 0
    cab['speed'] = 0
    f_time = lambda x: x.UNIX_time2-x.UNIX_time1
    f_speed = lambda x: int(x.dist/x.time+0.5)
    cab['time'] = cab.apply(f_time,axis=1)
    cab['dist'] = cab.apply(calculate_dist,axis=1)
    cab['speed'] = cab.apply(f_speed,axis=1)
    print(cab.head(10))
    save_file_name = output_path + 'all_chosed01.csv'
    cab.to_csv(save_file_name, sep=',', index=False,
                   columns=['cab_id',
                            'latitude1',
                            'longitude1', 'fare1',
                            'UNIX_time1',
                            'latitude2', 'longitude2',
                            'fare2',
                            'UNIX_time2', 'status',
                            'sequence', 's_node', 'e_node', 'index','time','dist','speed'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('计算每条记录的时间距离和速度')
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def per_se_calculate(input_path, output_path):
    # 按照每个文件进行处理
    start_time = time.time()
    files = os.listdir(input_path)
    count_of_files = len(files)
    print(count_of_files)

    file_id = 0  # 文件编号从0开始
    for i in files:
        # 取出源文件名
        source_file_name = re.findall(r"(.+?).csv", files[file_id])  # 正则化取出起点-终点名
        print('第%d个文件' % file_id)
        print(source_file_name[0])
        source_file = input_path + files[file_id]
        print(source_file)

        # 读入数据
        cab_df = pd.read_csv(source_file)
        cab_df = cab_df.apply(pd.to_numeric, downcast='signed')
        # print("输出 cab_df")
        # print(cab_df.head(5))
        # print(cab_df.tail(5))
        cab_df_gp =cab_df.groupby('index')
        se_info = pd.DataFrame(columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                           'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                           'tot_angle', 'angle', 'max_angle', 'min_angle'])

        for group in cab_df_gp:
            # 计算角度
            one_traj = group[1]

            s_df = one_traj.head(1)
            e_df = one_traj.tail(1)
            # print(s_df)
            # print(e_df)
            se_long_add = int(e_df['longitude1']) - int(s_df['longitude2'])
            se_lati_add = int(e_df['latitude1']) - int(s_df['latitude2'])
            # print(se_long_add)
            # print(se_lati_add)
            # calculate the angle of rotation
            se_angle = math.atan2(se_lati_add, se_long_add)
            se_angle = int(se_angle * 180 / math.pi)
            one_traj.loc[:,'angle'] = one_traj.apply(calculate_angle, axis=1, **{'angle1': se_angle})  # {'angle1':se_angle}
            one_traj.loc[:,'time'] = 0
            one_traj.loc[:,'dist'] = 0
            one_traj.loc[:,'speed'] = 0
            f_time = lambda x: abs(x.UNIX_time2 - x.UNIX_time1)
            f_speed = lambda x: int(x.dist / x.time + 0.5)
            one_traj.loc[:,'time'] = one_traj.apply(f_time, axis=1)
            one_traj.loc[:,'dist'] = one_traj.apply(calculate_dist, axis=1)
            one_traj.loc[:,'speed'] = one_traj.apply(f_speed, axis=1)

            # 按照起点和终点存储每条轨迹的特征
            # 特征包括：总时间/总距离/平均速度/最大速度/最小速度/总角度/平均角度/最大角度/最小角度
            cab_id = int(s_df['cab_id'])
            sequence = int(s_df['sequence'])
            s_node = int(s_df['e_node'])
            e_node = int(e_df['s_node'])
            index = int(s_df['index'])
            tot_time = one_traj[one_traj['status'].isin([3])]['time'].sum()
            dist = one_traj[one_traj['status'].isin([3])]['dist'].sum()
            speed = int(dist/tot_time+0.5)
            max_speed = one_traj[one_traj['status'].isin([3])]['speed'].max()
            min_speed = one_traj[one_traj['status'].isin([3])]['speed'].min()
            tot_angle = one_traj[one_traj['status'].isin([3])]['angle'].sum()
            angle = int(one_traj[one_traj['status'].isin([3])]['angle'].mean()+0.5)
            max_angle = one_traj[one_traj['status'].isin([3])]['angle'].max()
            min_angle = one_traj[one_traj['status'].isin([3])]['angle'].min()
            se_info = se_info.append(
                {'cab_id':cab_id, 'sequence':sequence, 's_node':s_node, 'e_node':e_node, 'index':index,
                                           'time':tot_time, 'dist':dist, 'speed':speed, 'max_speed':max_speed, 'min_speed':min_speed,
                                           'tot_angle':tot_angle, 'angle':angle, 'max_angle':max_angle, 'min_angle':min_angle}, ignore_index=True)

        mid_dist = int(se_info['dist'].median())//1000
        se_info['mid_dist'] = mid_dist
        se_info = se_info.apply(pd.to_numeric, downcast='signed')
        if mid_dist < 3:
            save_file_name = output_path + '0-3/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                       columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                           'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                           'tot_angle', 'angle', 'max_angle', 'min_angle','mid_dist'])
        elif mid_dist < 5:
            save_file_name = output_path + '3-5/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        elif mid_dist < 7:
            save_file_name = output_path + '5-7/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        elif mid_dist < 10:
            save_file_name = output_path + '7-10/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        elif mid_dist < 15:
            save_file_name = output_path + '10-15/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        elif mid_dist < 20:
            save_file_name = output_path + '15-20/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        elif mid_dist >= 20:
            save_file_name = output_path + '20+/' + source_file_name[0] + '.csv'
            se_info.to_csv(save_file_name, sep=',', index=False,
                           columns=['cab_id', 'sequence', 's_node', 'e_node', 'index',
                                    'time', 'dist', 'speed', 'max_speed', 'min_speed',
                                    'tot_angle', 'angle', 'max_angle', 'min_angle', 'mid_dist'])
        file_id += 1  # 文件编号自加1

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('计算每条记录的时间距离和速度')
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def calculate_angle(row, angle1):
   dx2 = int(row['longitude2']) - int(row['longitude1'])
   dy2 = int(row['latitude2']) - int(row['latitude1'])
   angle2 = math.atan2(dy2, dx2)
   angle2 = int(angle2 * 180/math.pi)
   # print(angle2)
   if angle1*angle2 >= 0:
       included_angle = abs(angle1-angle2)
   else:
       included_angle = abs(angle1) + abs(angle2)
       if included_angle > 180:
           included_angle = 360 - included_angle
   return included_angle

