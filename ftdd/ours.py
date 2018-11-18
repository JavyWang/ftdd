import time
import pandas as pd


def cab_coordinate(inputpath,outputpath,rownumber):
    #功能：把整合完的文件中的经纬度换算为方格的坐标，rownumber代表方格的行数（列数），默认行列数相同
    #顺序1
    start_time = time.time()
    df = pd.read_csv(inputpath)
    end_time = time.time()
    print(end_time-start_time)
    print(df.head())
    #latitude:2000-10000   区间长度：8000   一格40
    #longitude:0-6000   区间长度：6000   一格30   100*100
    unit_latitude = int(8000/rownumber)
    unit_longitude = int(6000/rownumber)
    f_latitudeid = lambda x:(x-2000)/unit_latitude+1
    f_longitudeid = lambda x:x/unit_longitude+1
    start_time = time.time()
    df['latitude1'] = df['latitude1'].map(f_latitudeid)
    df['latitude2'] = df['latitude2'].map(f_latitudeid)
    df['longitude1'] = df['longitude1'].map(f_longitudeid)
    df['longitude2'] = df['longitude2'].map(f_longitudeid)
    df[['latitude1','latitude2','longitude1','longitude2']] = df[['latitude1','latitude2','longitude1','longitude2']].astype(int)
    df = df.apply(pd.to_numeric, downcast='signed')
    end_time = time.time()
    print(end_time-start_time)
    print(df)
    start_time = time.time()
    f_id = lambda longitude,latitude:longitude+latitude*rownumber
    id1 = map(f_id,df['longitude1'],df['latitude1'])
    id2 = map(f_id,df['longitude2'],df['latitude2'])
    df['id1'] = list(id1)
    df['id2'] = list(id2)
    df = df.apply(pd.to_numeric, downcast = 'unsigned')
    df.to_csv(outputpath,sep=',', index=False,
                                columns=['cab_id', 'id1',
                                         'fare1', 'UNIX_time1',
                                         'id2','fare2',
                                         'UNIX_time2', 'status', 'sequence'])
    end_time = time.time()
    print(end_time-start_time)
    print(df)


def edge_receiver(inputpath,outputpath,transformer):
    #功能：将换算为方格坐标文件，并将（4754，4621换算成47544621）其转化为路径，统计每条路径的频次存储到outputpath所指文件中
    #顺序2
    #需要手动将outputpath存好的csv文件改成txt类型以备之后的函数使用
    df = pd.read_csv(inputpath)
    f_edge = lambda x,y:x*transformer+y
    edges = list(map(f_edge,df['id1'],df['id2']))
    df['path'] = edges
    print(edges)
    result = pd.value_counts(edges)
    print(result)
    print(type(result))
    result.to_csv(outputpath,sep=',')

def driver_path(inputpath,outputpath,transformer):
    #功能：将换算为方格坐标的文件转换成路径图，并用一个int代表一段向量，方法为将起始点乘以transformer（根据总方格数设置）
    #顺序3
    df = pd.read_csv(inputpath)
    f_edge = lambda x,y:x*transformer+y
    edges=list(map(f_edge,df['id1'],df['id2']))
    df['path'] = edges
    df = df.apply(pd.to_numeric, downcast='signed')
    #df.sort_values(by=['path'],inplace=True)
    df.to_csv(outputpath,sep=',',index=False,columns=['cab_id','fare1','UNIX_time1','fare2','UNIX_time2','status','sequence','path'])

def frequency_index(path,edges,frequency):
    #功能：driver_score中的map函数，用于获取index
    Index = edges.index(path)
    return frequency[Index]
def driver_score(inputpath1,inputpath2,outputpath1):

    #功能：统计每条相邻路径的频次
    #inputpath1为统计好的：边--频次文件
    #inputpath2为换算成方格坐标并带有path的文件
    #顺序4
    df_edge = pd.read_table(inputpath1,sep=',',names=['edges','frequency'])
    edges = list(df_edge['edges'])
    frequency = list(df_edge['frequency'])

    startime = time.time()
    inputpath2 = '/Users/ntldr/Desktop/cabspottingid/all_cab_trace_only_fare_path200.csv'
    df_driver = pd.read_csv(inputpath2)
    freq_list = list(map(frequency_index,df_driver['path'],edges,frequency))
    df_driver['frequency'] = freq_list
    df_driver.to_csv(outputpath1,sep=',',index=False,columns=['cab_id','fare1','UNIX_time1','fare2','UNIX_time2','status','sequence','path','frequency'])
    endtime = time.time()
    print(endtime-startime)

def score_calculate(inputpath1):
    #功能：将处理好的带有对应频次的文件作图，并计算每个司机的平均频次
    #顺序5
    df = pd.read_csv(inputpath1)
    startime = time.time()
    data = df.groupby('cab_id')['frequency'].mean()
    endtime = time.time()
    print(endtime-startime)
    print(data)
    #print(data[2])
    #print(type(data))
    print(list(data))
    a = list(data)
    plt.hist(a,bins=20, density=0, facecolor='blue', edgecolor='black',alpha=1)
    plt.savefig('20.png') #默认格式用多少bin来存储表示
    plt.show()
