"""
  2018-10-14 @Javy Wang
  note: 功能放到主函数里面执行
"""
from ftdd.utils import *
# from ftdd.models import *
from ftdd.fdfds2011 import *
from ftdd.new import *


def main():
    """
    需要执行的功能应该在这里实现，通过调用不同的函数实现不同的功能

    :return:
    """
    # Caution！ 这里文件路径要设置为自己数据所在的文件目录
    data_path = {
        'data1': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting1\\',
        'data2': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting2\\',
        'data3': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting3\\',
        'data4': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting4\\',
        'data5': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting5\\',
        'data6': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting6\\',
        'data7': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting7\\',
        'data9': 'D:\MyWorks\Researches\TaxiFraudDetection\codes\\ftdd\\ftdd\data\\cabspotting9\\',
        'data10': '/Users/javy/PycharmProjects/ftdd/ftdd/data/cabspotting10/',
        'data11': '/Users/javy/PycharmProjects/ftdd/ftdd/data/cabspotting10/new/'

    }
    caculate(data_path['data11'], data_path['data11'])
    # get_interesting_sites(data_path['data10'], data_path['data10'], 100)
    # get_nodes_label(data_path['data10'], data_path['data10'], 8)
    # get_node_pair(data_path['data10'], data_path['data10'])
    # find_trajectory(data_path['data10'], data_path['data10'])  # this step is the most consumption
    # find_fraud_trajectory(data_path['data10'], data_path['data10'])
    # calculate_fraud_degree(data_path['data10'], data_path['data10'])
    # print(stat_dict)

    # data_transfer(data_path['data4'], data_path['data6'])
    # data_concat(data_path['data9'], data_path['data10'])

    # grid_m = 20             # the number of rows in road grid
    # grid_n = 50             # the number of columns in road grid
    # long_base = -122.42     # the start longitude
    # lati_base = 37.75       # the start latitude
    # long_term = -122.40     # the end longitude
    # lati_term = 37.80       # the end latitude
    # long_gap = 0.001        # the longitude space between two nodes at the same longitude
    # lati_gap = 0.001        # the latitude space between two nodes at the same latitude
    # input_file = data_path['data5'] + 'net_data.csv'
    # # data = data_filter(input_file, long_base, lati_base, long_term, lati_term)
    # data = pd.read_csv(input_file)    # build a dataframe to cover all cab traces
    # road_graph = RoadGraph(grid_m, grid_n, long_base, lati_base, long_term, lati_term, long_gap, lati_gap, data)
    # road_graph.get_graph()      # create the road graph


if __name__ == '__main__':
    main()
