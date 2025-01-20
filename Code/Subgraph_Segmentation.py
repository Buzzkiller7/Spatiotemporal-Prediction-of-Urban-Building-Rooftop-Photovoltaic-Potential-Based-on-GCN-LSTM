import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm


# 定义一个函数，用于将子图和相应的太阳能数据保存为CSV文件
def save_subgraphs_and_solar_data(adjacency_list_df, solar_data_df, output_directory, min_nodes=1):
    # 定义深度优先搜索（DFS）函数来找到所有连接的节点
    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs(graph, neighbor, visited)
        return visited

    def create_graph(adj_list):
        graph = defaultdict(set) # 使用defaultdict来创建一个空的图
        for _, row in adj_list.iterrows(): 
            graph[row['shadowing_building']].add(row['shadowed_building'])
            graph[row['shadowed_building']].add(row['shadowing_building'])
        return graph

    def split_graph_into_subgraphs(adj_list, min_nodes):
        graph = create_graph(adj_list)
        visited_nodes = set()  # visited_nodes是一个集合，用于存储已访问的节点
        subgraphs = []
        current_subgraph_nodes = set()

        # 对每个节点执行DFS，找到连通的子图，并确保每个输出CSV的节点数大于等于阈值
        for node in graph:
            if node not in visited_nodes:
                subgraph = dfs(graph, node)
                visited_nodes.update(subgraph)
                if len(current_subgraph_nodes) + len(subgraph) >= min_nodes:
                    subgraphs.append(current_subgraph_nodes.union(subgraph))
                    current_subgraph_nodes = set()
                else:
                    current_subgraph_nodes.update(subgraph)

        # 确保最后一组节点也被添加到子图中，即使它们的数量没有达到阈值
        if current_subgraph_nodes:
            subgraphs.append(current_subgraph_nodes)

        return subgraphs

    # 转置sub_solar_data_df
    def transpose(df):
        columns_to_drop = [
            'OID_', 'use', 'age', 'relhmax', 'STORY', 'Shape_Area', 'GFA', 'Shape_Length',
            'BuildingPerimeterShapeFactor', 'BuildingVolumeShapeFactor', 'Volume', 'Slope', 'BO'
        ]
        df_dropped = df.drop(columns=columns_to_drop)
        df_transposed = df_dropped.T
        return df_transposed

    # 使用上述函数将邻接列表分割为子图
    subgraphs = split_graph_into_subgraphs(adjacency_list_df, min_nodes)

    # 保存子图信息
    subgraphs_info = []
    # 对每个子图，提取相关数据并保存为CSV
    for index, subgraph in tqdm(enumerate(subgraphs), desc='subgraph', unit='subgraph'):
        # 提取与当前子图相对应的邻接列表行
        sub_adj_list_df = adjacency_list_df[adjacency_list_df['shadowing_building'].isin(subgraph) |
                                            adjacency_list_df['shadowed_building'].isin(subgraph)] # 提取与当前子图相对应的邻接列表行
        
        # 提取与当前子图相对应的太阳能数据行
        sub_solar_data_df = solar_data_df[solar_data_df['OID_'].isin(subgraph)]
        
        # 转置sub_solar_data_df为buildings_Roof_Solar_transposed_index.csv，以便每行对应一个时间戳下的所有节点的太阳能数据
        sub_solar_data_transposed_df = transpose(sub_solar_data_df)

        # 保存子图和太阳能数据
        sub_adj_list_path = f"{output_directory}/sub_adj_list_{index}.csv"
        sub_solar_data_path = f"{output_directory}/sub_solar_data_{index}.csv"
        sub_solar_data_transposed_path = f"{output_directory}/sub_solar_data_transposed_{index}.csv"
        sub_adj_list_df.to_csv(sub_adj_list_path, index=False) 

        # 判断sub_solar_data_df是否为空
        if sub_solar_data_df.empty:
            print (f"sub_solar_data_df is empty, index = {index}")
        sub_solar_data_df.to_csv(sub_solar_data_path, index=False)
        sub_solar_data_transposed_df.to_csv(sub_solar_data_transposed_path, index=False)
        subgraphs_info.append([index, len(subgraph)])

    subgraphs_info_df = pd.DataFrame(subgraphs_info, columns=['subgraph_index', 'subgraph_size'])
    subgraphs_info_df.to_csv(f"{output_directory}/subgraphs_info_new.csv", index=False)


season_list = ['spring', 'summer', 'autumn']
# season_list = ['winter']
for season in tqdm(season_list, desc='season', unit='season'):
    print('importing data...')
    adjacency_list_df = pd.read_csv(f'D:\\Desktop\\Data\\adjacency_list_season\\buildings_adjacency_list_{season}_valid_dis.csv')
    print('data-1 imported')
    solar_data_df = pd.read_csv(fr'D:\Desktop\Data\Roof_Solar\buildings_Roof_Solar_{season}_19to22.csv')
    print('data-2 imported')
    output_directory = fr'D:\Desktop\Data\subgraphs_and_solar_data\{season}'
    # if output_directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    save_subgraphs_and_solar_data(adjacency_list_df, solar_data_df, output_directory, min_nodes=1)
    print('data saved')