"""
  2018-10-16 @Javy Wang
  note: this module is used to repeat their methods
  reference: A Taxi Driving Fraud Detection System (2011)
"""
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
import os
import re
import time
import matplotlib.pyplot as plt
import math


# this method is designed for dataframe
def get_interesting_sites(input_path, output_path, num):

    # load data
    grid_file = input_path + 'grid.csv'  # 文件名
    grid_df = pd.read_csv(grid_file)  # all trajectories with fare
    grid_df.sort_values(by=['weight'], ascending=False, inplace=True)
    inte_sites = grid_df.head(num)   # take the top num data
    f = lambda x: pd.Series([int((x.grid_id % 30)*200*8.76), int((x.grid_id // 30)*160*11.12)], index=['lo_x', 'la_y'])
    inte_sites[['lo_x', 'la_y']] = inte_sites.apply(f, axis=1)
    save_file_name = output_path + 'interesting_sites.csv'  # 文件名
    inte_sites.to_csv(save_file_name, sep=',', index=False, columns=['grid_id', 'pick_up', 'drop_off', 'weight', 'lo_x', 'la_y'])
    # return inte_sites


# this method is designed for dataframe
def get_nodes_label(input_path, output_path, num):
    """
    find num pairs of source and end nodes which have moderate distance
    :param input_path:
    :param num:
    :param output_path:
    :return:
    """
    # use k-means to cluster these interesting_sites
    # load data
    sites_file = input_path + 'interesting_sites.csv'  # 文件名
    grid_df = pd.read_csv(sites_file)  # all trajectories with fare
    data = grid_df[['lo_x', 'la_y']].values.tolist()

    clf = KMeans(num)  # num clusters, clf is a instance of KMeans
    clf.fit(data)  # 加载数据集合
    labels = clf.labels_
    print(labels, type(labels))  # 显示 the cluster of each node
    print(clf.inertia_)  # 显示聚类效果
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(len(data)):
        plt.plot(data[i][0], data[i][1], mark[clf.labels_[i]])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质点，用特殊图型
    centroids = clf.cluster_centers_  # take the center of each cluster
    print(centroids)
    for i in range(num):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize=12)

    plt.gca().invert_xaxis()
    plt.show()

    grid_df['labels'] = labels   # add the labels
    save_file_name = output_path + 'node_labels01.csv'  # 文件名
    grid_df.to_csv(save_file_name, sep=',', index=False, columns=['grid_id', 'pick_up', 'drop_off', 'weight', 'lo_x', 'la_y', 'labels'])


def get_node_pair(input_path, output_path):
    # load data
    sites_file = input_path + 'node_labels01.csv'  # 文件名
    grid_df = pd.read_csv(sites_file)  # all trajectories with fare
    grid_df = grid_df[grid_df['weight']>400]
    grid_df_gp = grid_df.groupby('labels')
    nodes_list = []   # save the grid list which is the selected source and end node pairs
    cluster_top = []   # save top grids in each cluster
    for group in grid_df_gp:
        each_cluster = group[1]
        each_cluster.sort_values(by=['weight'])
        each_cluster.index = range(1, len(each_cluster)+1)
        cluster_count = each_cluster['weight'].count()
        cluster_top.append(each_cluster.loc[1, 'grid_id'])
        if cluster_count > 2:
            for g_id in range(2, cluster_count+1):
                nodes_list.append([each_cluster.loc[1, 'grid_id'], each_cluster.loc[g_id, 'grid_id']])

        # if cluster_count > 3:
        #     for g_id1 in range(3, cluster_count+1):
        #         nodes_list.append([each_cluster.loc[2, 'grid_id'], each_cluster.loc[g_id1, 'grid_id']])

    for i in range(0, len(cluster_top)):
        for j in range(i+1, len(cluster_top)):
            nodes_list.append([cluster_top[i], cluster_top[j]])

    nodes_df = pd.DataFrame(nodes_list, columns=['source_grid', 'end_grid'])
    print(len(nodes_df))
    save_file_name = output_path + 'nodes_pairs.csv'
    nodes_df.to_csv(save_file_name, sep=',', index=False, columns=['source_grid', 'end_grid'])
    # return nodes_list
    # select the top nodes with high weight in per cluster(there are many choices)


# this method is designed for graph
def find_interesting_sites(graph, num):
    """
    find top num nodes which have the highest weight in the graph

    :param graph:
    :param num:
    :return: a series contain the top num nodes with attributes
    """
    node_attr_dict = nx.get_node_attributes(graph, 'WEIGHT')
    print(type(node_attr_dict))
    print(node_attr_dict)
    node_attr_series = pd.Series(node_attr_dict)
    node_attr_series.sort_values(ascending=False, inplace=True)
    interest_sites = node_attr_series[:num]
    print(interest_sites)
    print(interest_sites.index[0])
    return interest_sites


# this method is designed for graph
def select_nodes_pairs(graph, num, sites):
    """
    find num pairs of source and end nodes which have moderate distance
    :param graph:
    :param num:
    :param sites:
    :return:
    """
    # use k-means to cluster these interesting_sites
    data = []
    for idx in range(0, len(sites)):
        data.append([graph.nodes[sites.index[idx]]['latitude'], graph.nodes[sites.index[idx]]['longitude']])

    print(data)

    clf = KMeans(num)  # num clusters, clf is a instance of KMeans
    clf.fit(data)  # 加载数据集合
    labels = clf.labels_
    print(labels, type(labels))  # 显示 the cluster of each node
    print(clf.inertia_)  # 显示聚类效果
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(len(data)):
        plt.plot(data[i][0], data[i][1], mark[clf.labels_[i]])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质点，用特殊图型
    centroids = clf.cluster_centers_  # take the center of each cluster
    print(centroids)
    for i in range(num):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize=12)

    plt.show()

    # select source and end nodes between different clusters
    # select the top nodes with high weight in per cluster


def find_trajectory(input_path, output_path):
    """
    find all trajectory between a pair of source and end nodes

    :param input_path: each cab trace file
    :param output_path: a dataframe cover all trajectories between these node pairs
    :return:
    """
    # create a dataframe to save all trajectories between these node pairs
    start_time = time.time()
    traj_df = pd.DataFrame(columns=('cab_id', 'sequence', 's_node', 'e_node', 'time', 'dist'))

    # load data
    source_file = input_path + 'all_cab_trace_only_fare.csv'
    cab_trace_df = pd.read_csv(source_file)
    print("输出 cab_trace_df")
    print(cab_trace_df.head())
    nodes_file = input_path + 'nodes_pairs.csv'
    nodes_df = pd.read_csv(nodes_file)
    sites_pair = nodes_df.values.tolist()

    # transfer the data type of cab_trace_df
    cab_trace_df = cab_trace_df.apply(pd.to_numeric, downcast='signed', errors='ignore')

    # find the num of trajectories in cab_trace_df
    cab_trace_group = cab_trace_df.groupby(['cab_id', 'sequence'])

    # go through all group
    for group in cab_trace_group:
        # take out the slice of the traj_id trajectory
        # print(group)



        per_traj_df = group[1]
        s_node_df = per_traj_df.loc[(per_traj_df['status'].isin(['1']))]
        e_node_df = per_traj_df.loc[(per_traj_df['status'].isin(['2']))]
        if s_node_df.empty is False and e_node_df.empty is False:
            node_pair = coordinates_to_grid(s_node_df, e_node_df)
            if node_pair in sites_pair:
                s_node = node_pair[0]
                e_node = node_pair[1]
                cab_id = int(s_node_df.head(1)['cab_id'])
                traj_id = int(s_node_df.head(1)['sequence'])
                traj_time = int(e_node_df.head(1)['UNIX_time1']) - int(s_node_df.head(1)['UNIX_time2'])
                traj_dist = get_dist(per_traj_df)
                # add this trajectory to traj_df
                traj_df = traj_df.append({'cab_id': cab_id, 'sequence': traj_id, 's_node': s_node, 'e_node': e_node, 'time': traj_time, 'dist': traj_dist}, ignore_index=True)
                print(traj_df.tail())

    save_file_name = output_path + 'selected_trajectories.csv'  # 文件名
    traj_df.to_csv(save_file_name, sep=',', index=False, columns=['cab_id', 'sequence', 's_node', 'e_node', 'time', 'dist'])

    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数


def coordinates_to_grid(s_df, e_df):
    """
    get the index of start node and end node

    :param s_df:
    :param e_df:
    :return:
    """
    longitude_base = 0
    latitude_base = 2000
    longitude_gap = 200
    latitude_gap = 160
    grid_m = 30

    s_latitude = s_df['latitude2'].values[0]
    s_longitude = s_df['longitude2'].values[0]
    e_latitude = e_df['latitude1'].values[0]
    e_longitude = e_df['longitude1'].values[0]

    # coz the index of first node is 1 so that 1 should be added following
    s_grid_id = int((s_longitude - longitude_base) // longitude_gap) + 1 + int(
        (s_latitude - latitude_base) // latitude_gap) * grid_m
    e_grid_id = int((e_longitude - longitude_base) // longitude_gap) + 1 + int(
        (e_latitude - latitude_base) // latitude_gap) * grid_m

    return [s_grid_id, e_grid_id]


def get_dist(per_traj):
    """
    calculate the distance between source and end node
    :param per_traj:
    :return:
    """

    traj_dist_df = per_traj.loc[(per_traj['status'].isin(['3']))]
    traj_dist_df['dist'] = traj_dist_df.apply(calculate_dist, axis=1)
    whole_dist = traj_dist_df['dist'].sum()
    return whole_dist


def calculate_dist(one_traj):
    # calculate the distance between Adjacent two points
    lat_dist = abs(one_traj['latitude1'] - one_traj['latitude2'])*11.12
    lon_dist = abs(one_traj['longitude1'] - one_traj['longitude2'])*8.76
    dist = int(math.sqrt(pow(lat_dist, 2) + pow(lon_dist, 2)))
    return dist


def find_fraud_trajectory(input_path, output_path):
    """
    find fraud trajectory between each pair of source and end nodes
    :param input_path:
    :param output_path:
    :return:
    """
    # load data
    start_time = time.time()
    traj_file = input_path + 'selected_trajectories.csv'  # 文件名
    traj_df = pd.read_csv(traj_file)  # all trajectories between all pairs of source and end nodes


    # go through each trajectory between each pair of source and end nodes
    traj_group = traj_df.groupby(['s_node', 'e_node'], sort=True)
    fraud_df = pd.DataFrame(columns=('cab_id', 'sequence', 's_node', 'e_node', 'time', 'dist', 'time_norm', 'dist_norm', 'score', 'fraud'))
    for group in traj_group:
        # take out the slice of the trajectory for each source and end node
        # print(group)
        slice_df = group[1]
        time_min = slice_df['time'].min()
        time_max = slice_df['time'].max()
        f_time = lambda x: round((x.time - time_min) / (time_max - time_min), 4)
        slice_df['time_norm'] = slice_df.apply(f_time, axis=1)
        dist_min = slice_df['dist'].min()
        dist_max = slice_df['dist'].max()
        f_dist = lambda x: round((x.dist - dist_min) / (dist_max - dist_min), 4)
        slice_df['dist_norm'] = slice_df.apply(f_dist, axis=1)
        slice_df['score'] = slice_df.apply(lambda x: x.time_norm + x.dist_norm, axis=1)
        slice_df.sort_values(by='score', ascending=False, inplace=True)   # descending
        slice_df.index = range(1, len(slice_df) + 1)
        slice_df['fraud'] = 0
        if slice_df['score'].count() > 10:
            slice_df.loc[1:10, 'fraud'] =1
            slice_df.loc[11:, 'fraud'] = 0
        else:
            fraud_num = int(0.3 * slice_df['score'].count())
            if fraud_num > 1:
                slice_df.loc[1:fraud_num, 'fraud'] = 1
                slice_df.loc[fraud_num+1:, 'fraud'] = 0

        fraud_df = pd.concat([fraud_df, slice_df])

    # save dataframe
    save_file_name = output_path + 'fraud_trajectories.csv'  # 文件名
    fraud_df.to_csv(save_file_name, sep=',', index=False,
                   columns=['cab_id', 'sequence', 's_node', 'e_node', 'time', 'dist', 'time_norm', 'dist_norm', 'score', 'fraud'])
    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数
    # return fraud_df


def calculate_fraud_degree(input_path, output_path):
    """
    calculate the fraud degree of each driver according the selected trajectory

    :param input_path:
    :param output_path:
    :return:
    """
    # load data
    fraud_file = input_path + 'fraud_trajectories.csv'  # 文件名
    fraud_df = pd.read_csv(fraud_file)  # all trajectories between all pairs of source and end nodes

    start_time = time.time()

    # go through each cab to count the fraud trajectory number and then
    traj_group = fraud_df.groupby(['cab_id'], sort=True)
    cab_score = pd.DataFrame(columns=('cab_id', 'traj_num', 'fraud_traj_num', 'score'))
    for group in traj_group:
        each_cab_df = group[1]
        each_cab_df.index = range(1, len(each_cab_df) + 1)
        cab_id = each_cab_df.loc[1, 'cab_id']
        traj_num = each_cab_df['fraud'].count()
        fraud_traj_num = each_cab_df['fraud'][each_cab_df['fraud']>0].count()
        score = round(fraud_traj_num / traj_num, 3)

        cab_score = cab_score.append({'cab_id': cab_id, 'traj_num': traj_num, 'fraud_traj_num': fraud_traj_num, 'score': score}, ignore_index=True)

    save_file_name = output_path + 'cab_fraud_degree.csv'  # 文件名
    cab_score.to_csv(save_file_name, sep=',', index=False,
                    columns=['cab_id', 'traj_num', 'fraud_traj_num', 'score'])
    end_time = time.time()
    cost_time = int(end_time - start_time) / 60  # 计算用时
    print('%.2f ' % cost_time)  # 显示用时，保留两位小数




