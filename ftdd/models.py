"""
  2018-10-14 @Javy Wang
  note: 这里将我们的网络模型构造一个类，其他的模型或者改进版的模型都可以从这个类继承
"""
import networkx as nx
import time
import pandas as pd


class RoadGraph:
    """
    这是个路网模型的基类，应当包含路网（网格）的构建，节点属性，边属性，及其他一些属性和功能，
    应当构建一些包含网络的统计特征的方法，绘图的方法

    """
    # Caution!!! __init__ not __int__
    def __init__(self, grid_m, grid_n, long_base, lati_base, long_term, lati_term, long_gap, lati_gap, input_data):
        self.grid_m = grid_m         # the number of rows in road grid
        self.grid_n = grid_n         # the number of columns in road grid
        self.long_base = long_base   # the base value of longitude
        self.lati_base = lati_base   # the base value of latitude
        self.long_term = long_term   # the terminal value of longitude
        self.lati_term = lati_term   # the terminal value of latitude
        self.long_gap = long_gap     # the gap  of longitude
        self.lati_gap = lati_gap     # the gap  of latitude

        self.input = input_data      # the input data--a dataframe of trace

    def get_node_list(self):
        """
        A function to create nodes in network

        :return:
            node_list: the list of nodes
            node_attr: the attributes of nodes
        """
        node_list = []  # 创建网络的节点列表
        node_attr = {}  # 创建网络的节点属性字典
        for i in range(1, self.grid_m * self.grid_n + 1):
            node_list.append(i)
            node_attr[i] = dict(WEIGHT=0,
                                PICKUP=0,
                                DROPOFF=0,
                                )
        return node_list, node_attr

    def get_edge_list(self, node_attr):
        """
        A function to create edges, edges' attributes and nodes' attributes in network

        :param node_attr:
        :return edge_list:
                edge_attr:
                node_attr:
        """
        start_time = time.time()

        trace_df = self.input
        trace_df.index = range(1, len(trace_df) + 1)  # 行索引从1开始，即节点编号从1开始
        print("输出 trace_df")
        print(trace_df)

        edge_list = []  # 创建网络的连边列表
        edge_attr = {}  # 创建网络的节点属性字典
        for i in range(1, len(trace_df) + 1):
            # get the index of start node and end node
            node_pair = self._coordinates_to_grid(trace_df, i)

            # 确定是否是一条新边，是的话添加边及其属性，否则更新边属性
            if node_pair in edge_list:
                edge_attr[node_pair]['WEIGHT'] += 1
            else:
                edge_list.append(node_pair)
                edge_attr[node_pair] = dict(WEIGHT=1)

            # update nodes' attributes
            node_attr[node_pair[0]]['WEIGHT'] += 1
            node_attr[node_pair[1]]['WEIGHT'] += 1
            if trace_df.loc[i, 'status'] == 1:
                node_attr[node_pair[1]]['PICKUP'] += 1
            elif trace_df.loc[i, 'status'] == 2:
                node_attr[node_pair[1]]['DROPOFF'] += 1

        end_time = time.time()
        cost_time = int(end_time - start_time) / 60  # 计算用时
        print('生成网络的边用时：%.2f 分钟' % cost_time)  # 显示用时，保留两位小数

        return edge_list, edge_attr, node_attr

    def _coordinates_to_grid(self, trace_df, index):
        """
        get the index of start node and end node
        :param index:
        :param trace_df:
        :return:
        """
        latitude1 = trace_df.loc[index, 'latitude1']
        longitude1 = trace_df.loc[index, 'longitude1']
        latitude2 = trace_df.loc[index, 'latitude2']
        longitude2 = trace_df.loc[index, 'longitude2']

        # coz the index of first node is 1 so that 1 should be added following
        grid_id1 = int((longitude1 - self.long_base) // self.long_gap) + 1 + int(
            (latitude1 - self.lati_base) // self.lati_gap) * self.grid_m
        grid_id2 = int((longitude2 - self.long_base) // self.long_gap) + 1 + int(
            (latitude2 - self.lati_base) // self.lati_gap) * self.grid_m

        grid_id = (grid_id1, grid_id2)
        return grid_id

    def get_graph(self):
        """
        路网创建应该包含以下操作：
            1. 节点操作：统计数据，添加节点属性；
            2. 边操作：统计数据，添加边属性；
        :return:
        """
        node_list, node_attr = self.get_node_list()
        edge_list, edge_attr, node_attr = self.get_edge_list(node_attr)   # 注意顺序！
        road_graph = nx.DiGraph()  # 得到有向图
        road_graph.add_edges_from(edge_list)
        nx.set_node_attributes(road_graph, node_attr)
        nx.set_edge_attributes(road_graph, edge_attr)
        nx.write_adjlist(road_graph, 'road_graph.adjlist')
        print("Output the road graph")
        print(road_graph)
        return road_graph

