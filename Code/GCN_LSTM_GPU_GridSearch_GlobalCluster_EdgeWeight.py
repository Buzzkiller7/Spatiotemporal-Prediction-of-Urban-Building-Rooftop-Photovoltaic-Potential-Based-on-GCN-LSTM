import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.model_selection import KFold

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置TensorFlow只分配所需的GPU内存空间,而不是分配所有的内存空间
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPUs available: {len(gpus)}')
    except RuntimeError as e:
        # 异常处理,可能在GPU已经被程序使用的情况下会发生
        print(e)

''' 函数和类定义 '''
# 定义图形信息类
class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], weights: np.ndarray, num_nodes: int):
        self.edges = edges 
        self.num_nodes = num_nodes
        self.weights = weights

# 定义图形卷积层
class GraphConv(layers.Layer):
    """
    这是一个GraphConv类,它是一个图卷积层,用于在图结构数据上进行卷积操作。这个类继承了TensorFlow的Layer类。

    类的初始化函数接收输入特征、输出特征、图形信息、汇总类型、结合类型和激活函数等参数。

    类中定义了以下几个主要方法:
    - aggregate:根据指定的汇总类型（"sum"、"mean"或"max"）对邻接节点的表示值进行汇总。
    - compute_nodes_representation:计算节点的表示值,通过将特征与权重矩阵进行矩阵乘法操作。
    - compute_aggregated_messages:计算汇总的消息,首先获取邻接节点的表示值,然后对其进行汇总,最后与权重矩阵进行矩阵乘法操作。
    - update:根据指定的结合类型（"concat"或"add"）更新节点的表示值。
    - call:这是类的主要方法,它首先计算节点的表示值,然后计算汇总的消息,最后更新节点的表示值。

    这个类是实现图卷积网络(GCN)的关键部分,它可以处理图结构的数据,对节点的特征进行卷积操作,从而学习节点的新的表示值。
    """
    def __init__(
        self, # self
        in_feat, # 输入特征
        out_feat, # 输出特征
        graph_info: GraphInfo, # 图形信息y
        aggregation_type="mean", # 汇总类型,可选值为 "sum"、"mean" 或 "max",汇总类型是如何汇总邻接节点的表示值的函数
        combination_type="concat", # 结合类型,可选值为 "concat" 或 "add",结合类型是值如何结合节点的表示值和邻接汇总信息的函数
        activation: typing.Optional[str] = None, # 激活函数,激活函数是指将节点的表示值转换为输出特征的函数
        **kwargs,
    ):
        super().__init__(**kwargs) # 调用父类的构造函数
        self.in_feat = in_feat # 输入特征
        self.out_feat = out_feat # 输出特征ky
        self.graph_info = graph_info # 图形信息
        self.aggregation_type = aggregation_type # 汇总类型
        self.combination_type = combination_type # 结合类型
        self.weight = tf.Variable( # 权重,权重是指将输入特征转换为输出特征的矩阵,它将在训练过程中学习
            initial_value=keras.initializers.glorot_uniform()( # 初始化权重
                shape=(in_feat, out_feat), dtype="float32" 
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation) # 激活函数,激活函数是指将节点的表示值转换为输出特征的函数

    def aggregate(self, neighbour_representations: tf.Tensor): # 汇总邻接节点的表示值
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func: # 如果汇总类型是 "sum"、"mean" 或 "max",则执行以下操作
            return aggregation_func( 
                neighbour_representations, # 邻接节点的表示值
                self.graph_info.edges[0], # 节点索引
                num_segments=self.graph_info.num_nodes, # 节点数
            ) 

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}") # 如果汇总类型不是 "sum"、"mean" 或 "max",则报错

    def compute_nodes_representation(self, features: tf.Tensor): 
        return tf.matmul(features, self.weight) # 矩阵相乘

    # 汇总邻接节点的表示值,确保处理的是有向图
    # def compute_aggregated_messages(self, features: tf.Tensor):
    #     neighbour_representations = tf.gather(features, self.graph_info.edges[1]) # 获取邻接节点的表示值, featuress
    #     aggregated_messages = self.aggregate(neighbour_representations)
    #     return tf.matmul(aggregated_messages, self.weight)
    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1]) # 邻居节点的特征
        edge_weights = tf.gather(self.graph_info.weights, self.graph_info.edges[1]) # 对应边的权重
        weighted_neighbour_representations = neighbour_representations * tf.expand_dims(edge_weights, -1) # 乘以权重
        aggregated_messages = self.aggregate(weighted_neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat": # 
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1) # 
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages 
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

# 定义LSTMGC类
class LSTMGC(layers.Layer): 
    def __init__(
        self, # self
        in_feat, # 输入特征
        out_feat, # 输出特征
        lstm_units: int, # LSTM单元数
        input_seq_len: int, # 输入序列长度
        output_seq_len: int, # 输出序列长度 
        graph_info: GraphInfo, # 图形信息
        graph_conv_params: typing.Optional[dict] = None, # 图形卷积参数
        **kwargs, # 其他参数
    ):
        super().__init__(**kwargs) # 调用父类的构造函数, 这里的父

        # 设置类属性
        self.in_feat = in_feat # 输入特征
        self.out_feat = out_feat # 输出特征
        self.lstm_units = lstm_units # LSTM单元数
        self.input_seq_len = input_seq_len # 输入序列长度
        self.output_seq_len = output_seq_len # 输出序列长度
        self.graph_info = graph_info # 图形信息
        self.graph_conv_params = graph_conv_params

        # 创建图形卷积层
        if graph_conv_params is None:
            graph_conv_params = { # 图形卷积参数
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params) # 创建图形卷积层

        self.lstm = layers.LSTM(lstm_units, activation="tanh") # 创建LSTM层
        
        self.dense = layers.Dense(output_seq_len) # 创建全连接Dense层, 对输入数据做线性变换:output = activation(dot(input, kernel) + bias)

    # 模型保存
    def get_config(self):
        config = super(LSTMGC, self).get_config()
        config.update({
            'in_feat': self.in_feat,
            'out_feat': self.out_feat,
            'lstm_units': self.lstm_units,
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            # 'graph_info': self.graph_info,  # 这里不能直接保存对象
            'graph_info': {'edges': self.graph_info.edges, 'num_nodes': self.graph_info.num_nodes},  # 将图信息转换为字典
            'graph_conv_params': self.graph_conv_params
        })
        return config

    @classmethod 
    def from_config(cls, config):
        # 重新创建 graph_info 对象
        graph_info = GraphInfo(edges=config['graph_info']['edges'], num_nodes=config['graph_info']['num_nodes'])
        config['graph_info'] = graph_info  # 替换回原来的对象
        return cls(**config)
    
    def call(self, inputs): 

        # 将输入的张量转置为 `(num_nodes, batch_size, input_seq_len, in_feat)`
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out张量形状为: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out) # 获取张量形状
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0], # 节点数
            shape[1], # 批次大小
            shape[2], # 输入序列长度
            shape[3], # 输出特征
        )

        # 将张量转换为形状为 `(batch_size * num_nodes, input_seq_len, out_feat)`
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out张量形状为: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense( 
            lstm_out
        )  # dense_output张量形状为: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # output张量形状为: (batch_size, output_seq_len, num_nodes)

# 定义简单LSTM模型
class SimpleLSTMModel(tf.keras.Model):
    def __init__(self, lstm_units, input_seq_len, num_features):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = layers.LSTM(lstm_units, activation='tanh', input_shape=(input_seq_len, num_features))
        self.dense = layers.Dense(num_features)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)

# 定义数据集创建函数
def create_tf_dataset(
    data_array: np.ndarray, # np.ndarray with shape `(num_time_steps, num_building)`
    input_sequence_length: int, # Length of the input sequence (in number of timesteps).
    forecast_horizon: int, # If `multi_horizon=True`, the target will be the values of the timeseries for 1 to `forecast_horizon` timesteps ahead. 
                           # If `multi_horizon=False`, the target will be the value of the timeseries `forecast_horizon` steps ahead (only one value).
    batch_size: int = 128, # Number of timeseries samples in each batch.
    shuffle=True, # Whether to shuffle output samples, or instead draw them in chronological order.
    multi_horizon=True, # See `forecast_horizon`.
    # Returns: A tf.data.Dataset instance.
):
    # 打包输入input
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )
    # 打包输出target
    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    ) # offset: 判断从哪里开始打包target
    target_seq_length = forecast_horizon if multi_horizon else 1 # target_seq_length: 判断一个target打包多少个值
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    # 将输入input和输出target打包成一个数据集
    dataset = tf.data.Dataset.zip((inputs, targets))

    # 判断是否打乱数据集
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()

# 定义LSTM数据集创建函数
def create_tf_dataset_for_lstm(
    data_array, 
    input_sequence_length, 
    forecast_horizon, 
    batch_size=128, 
    shuffle=True
):
    def process_batch(inputs, targets):
        # 输入数据的形状为 (batch_size, input_sequence_length, num_buildings, 1)
        inputs = tf.squeeze(inputs, axis=-1)  # 移除最后一个维度,形状变为 (batch_size, input_sequence_length, num_buildings)

        # 对于单步预测,我们只关注每个序列的下一个值
        targets = targets[:, 0, :]  # 选择每个序列的下一个时间步,形状变为 (batch_size, num_buildings)
        return inputs, targets

    # 创建原始数据集
    dataset = create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle)

    # 转换数据集以适应 LSTM 模型
    dataset = dataset.map(process_batch)

    return dataset

# 定义数据集划分函数, 将数据集划分为训练集和测试集
def split_dataset(data_array, train_size, test_dataset):
    num_time_steps = data_array.shape[0]
    num_train, num_test = (
        int(num_time_steps * train_size),
        int(num_time_steps * test_dataset),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    test_array = (data_array[num_train : (num_train + num_test)] - mean) / std

    return train_array, test_array

# 定义数据集创建函数
def create_datasets(input_sequence_length, forecast_horizon, train_array, val_array, batch_size=128, multi_horizon=False):
    train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size, multi_horizon=multi_horizon)
    val_dataset = create_tf_dataset(val_array, input_sequence_length, forecast_horizon, batch_size, multi_horizon=multi_horizon)

    train_dataset_lstm = create_tf_dataset_for_lstm(train_array, input_sequence_length, forecast_horizon, batch_size)
    val_dataset_lstm = create_tf_dataset_for_lstm(val_array, input_sequence_length, forecast_horizon, batch_size)

    return train_dataset, val_dataset, train_dataset_lstm, val_dataset_lstm

# 定义模型创建函数
def create_model(params, num_nodes, graph_info):
    lstm_gc_layer = LSTMGC(
        in_feat=params["in_feat"],
        out_feat=params["out_feat"],
        lstm_units=params["lstm_units"],
        input_seq_len=params["input_seq_len"],
        output_seq_len=params["forecast_horizon"] if params["multi_horizon"] else 1,
        graph_info=graph_info,
        graph_conv_params=params["graph_conv_params"]
    )

    input_layer = keras.Input(shape=(params["input_seq_len"], num_nodes, params["in_feat"]))
    output_layer = lstm_gc_layer(input_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model

# 创建简单LSTM模型
def create_simple_lstm(params, train_dataset_lstm, val_dataset_lstm):
    simple_lstm_model = SimpleLSTMModel(params["lstm_units"], params["input_seq_len"], 1)
    simple_lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="mean_squared_error", metrics=["mae"])

    return simple_lstm_model

# 网格搜索
def grid_search_gcn_lstm(params_grid, num_nodes, graph_info, train_array):
    best_model = None
    best_mae = float('inf')
    best_params = None
    best_history = None
    errors = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in params_grid:
        fold_mae = []
        
        for train_index, val_index in kf.split(train_array):
            train_data = train_array[train_index]
            val_data = train_array[val_index]
            train_dataset, val_dataset, _, _ = create_datasets(
                params["input_seq_len"], 1, train_data, val_data, 
                batch_size=params["batch_size"], multi_horizon=False
            )

            try:
                model = create_model(params, num_nodes, graph_info)
                history = model.fit(
                    train_dataset, 
                    validation_data=val_dataset,
                    epochs=params["epochs"], 
                    batch_size=params["batch_size"]
                )
                val_mae = min(history.history['val_mae'])
                fold_mae.append(val_mae)
            except Exception as e:
                errors.append((params, str(e)))
                continue

        avg_val_mae = np.mean(fold_mae)
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            best_model = model
            best_params = params
            best_history = history

    return best_model, best_params, best_history, errors

# 网格搜索
def grid_search_lstm(params_grid, train_array):
    best_model = None
    best_mae = float('inf')
    best_params = None
    best_history = None
    errors = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in params_grid:
        fold_mae = []

        for train_index, val_index in kf.split(train_array):
            train_data = train_array[train_index]
            val_data = train_array[val_index]
            _, _, train_dataset_lstm, val_dataset_lstm = create_datasets(
                params["input_seq_len"], 1, train_data, val_data, 
                batch_size=params["batch_size"], multi_horizon=False
            )

            try:
                model = create_simple_lstm(params, train_dataset_lstm, val_dataset_lstm)
                history = model.fit(
                    train_dataset_lstm, 
                    validation_data=val_dataset_lstm,
                    epochs=params["epochs"], 
                    batch_size=params["batch_size"]
                )
                val_mae = min(history.history['val_mae'])
                fold_mae.append(val_mae)
            except Exception as e:
                errors.append((params, str(e)))
                continue
            
        avg_val_mae = np.mean(fold_mae)
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            best_model = model
            best_params = params
            best_history = history

    return best_model, best_params, best_history, errors

# 定义评估函数
def calculate_metrics(y_true, y_pred, epsilon=1e-10):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    mmape = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))
    return mae, mse, rmse, mmape

# 定义评估模型函数
def evaluate_model(model, test_dataset, history, save_path, index):
    x_test, y = next(test_dataset.as_numpy_iterator())
    y_pred = model.predict(x_test)

    # 将 Tensor 转换为 NumPy 数组（如果它们不是 NumPy 数组）
    y_numpy = y.numpy() if hasattr(y, 'numpy') else y
    y_pred_numpy = y_pred if isinstance(y_pred, np.ndarray) else y_pred.numpy()

    # 计算指标
    mae, mse, rmse, mmape = calculate_metrics(y_numpy.mean(axis=-1), y_pred_numpy.mean(axis=-1))

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 创建三个子图

    # 可视化预测结果
    axs[0].plot(y_numpy.mean(axis=-1)[:, 0], label="Average Actual")
    axs[0].plot(y_pred_numpy.mean(axis=-1)[:, 0], label="Average Forecast")
    axs[0].legend()
    axs[0].set_title("Actual vs Forecast")

    # 可视化性能指标
    metrics = [mae, mse, rmse, mmape]
    labels = ["MAE", "MSE", "RMSE", "mMAPE"]
    axs[1].bar(labels, metrics, color='orange')
    for i, v in enumerate(metrics):
        axs[1].text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    axs[1].set_title("Performance Metrics")

    # 可视化损失函数
    axs[2].plot(history.history["loss"], label="Training Loss")
    axs[2].plot(history.history["val_loss"], label="Validation Loss")
    axs[2].legend()
    axs[2].set_title("Training and Validation Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'GCN+LSTM_{index}.png'))

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, mMAPE: {mmape}")
    return mae, mse, rmse, mmape

# 定义评估模型函数
def evaluate_model_lstm(model, test_dataset, history, save_path, index):
    x_test, y = next(iter(test_dataset))
    y_pred = model.predict(x_test)

    # 将 Tensor 转换为 NumPy 数组（如果它们不是 NumPy 数组）
    y_numpy = y.numpy() if hasattr(y, 'numpy') else y
    y_pred_numpy = y_pred if isinstance(y_pred, np.ndarray) else y_pred.numpy()

    # 计算指标
    mae, mse, rmse, mmape = calculate_metrics(y_numpy.mean(axis=-1), y_pred_numpy.mean(axis=-1))

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 创建三个子图

    # 可视化预测结果
    axs[0].plot(y_numpy.mean(axis=-1), label="Average Actual")
    axs[0].plot(y_pred_numpy.mean(axis=-1), label="Average Forecast")
    axs[0].legend()
    axs[0].set_title("Actual vs Forecast")

    # 可视化性能指标
    metrics = [mae, mse, rmse, mmape]
    labels = ["MAE", "MSE", "RMSE", "mMAPE"]
    axs[1].bar(labels, metrics, color='orange')
    for i, v in enumerate(metrics):
        axs[1].text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    axs[1].set_title("Performance Metrics")

    # 可视化损失函数
    axs[2].plot(history.history["loss"], label="Training Loss")
    axs[2].plot(history.history["val_loss"], label="Validation Loss")
    axs[2].legend()
    axs[2].set_title("Training and Validation Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'LSTM_{index}.png'))

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, mMAPE: {mmape}")
    return mae, mse, rmse, mmape

'''
代码的 main 主体部分
看我写的屎山。。。。。。。
太想吐了。。。。。。。 

数据集太大了似乎不能用4年的数据集, 先用两年的看看
- ( ) 需要再去把两年的数据集提取出来 21/3
'''
file_path = r"D:\Desktop\Data\subgraphs_and_solar_data"
subgraphs_info = pd.read_csv(file_path + r"\winter_subgraphs_info_labeled_updated_with_graph_structure.csv")
subgraphs_info = subgraphs_info [subgraphs_info['cluster_label'] != -1]
grouped_subgraphs = subgraphs_info.groupby('cluster_label')


#### 代码不同的部分
# save_path = fr"D:\Desktop\Data\subgraphs_and_solar_data\output\{season}"
# season_path = os.path.join(file_path, season)
season_path = os.path.join(file_path, 'global') 
# save_path = os.path.joun(file_path, 'output', season)
save_path = os.path.joun(file_path, 'edged_weighted_global_output')
if not os.path.exists(save_path):
    os.makedirs(save_path)

num_subgraphs = len([name for name in os.listdir(season_path) if name.startswith("sub_adj_list")])
# 储存评分
gcn_lstm_mae_list, gcn_lstm_mse_list, gcn_lstm_rmse_list, gcn_lstm_mmape_list = [0]*num_subgraphs, [0]*num_subgraphs, [0]*num_subgraphs, [0]*num_subgraphs
lstm_mae_list, lstm_mse_list, lstm_rmse_list, lstm_mmape_list = [0]*num_subgraphs, [0]*num_subgraphs, [0]*num_subgraphs, [0]*num_subgraphs

# for index in tqdm(range(num_subgraphs), desc='subgraph', unit='subgraph'):
for cluster_label, group in tqdm(grouped_subgraphs, desc='cluster', unit='cluster'):
    for group_index in tqdm(range(group.shape[0]), desc='subgraph', unit='subgraph'):
        index = group.iloc[group_index]['subgraph_index']
        adjacency_list = pd.read_csv(os.path.join(season_path, fr"sub_adj_list_{index}.csv")).to_numpy()
        edge_weights = pd.read_csv(fr"D:\Desktop\Data\subgraphs_and_solar_data\edge\sub_edge_attr_{index}.csv").to_numpy().squeeze()
        roof_solar_array = pd.read_csv(os.path.join(season_path, fr"sub_solar_data_transposed_{index}.csv")).to_numpy()
                
        # 数据预处理
        # 计算节点数,边数等
        num_nodes = len(set(adjacency_list[:, 0]).union(set(adjacency_list[:, 1])))
        num_edges = len(adjacency_list)
        adjacency_list = adjacency_list.astype(np.int32)

        # 创建GraphInfo实例
        node_indices, neighbor_indices = adjacency_list[:, 1], adjacency_list[:, 0]
        # graph = GraphInfo((node_indices.tolist(), neighbor_indices.tolist()), num_nodes)
        graph = GraphInfo((node_indices.tolist(), neighbor_indices.tolist()), edge_weights, num_nodes)

        # 数据集划分0.7训练集,其中训练集将拿去五折交叉验证,0.3测试集
        train_size, test_dataset = 0.7, 0.3
        train_array, test_array = split_dataset(roof_solar_array, train_size, test_dataset)

        # 保存GCN+LSTM模型的最佳参数
        best_params_gcn_lstm_list = []

        # 保存纯LSTM模型的最佳参数
        best_params_lstm_list = []

        # 创建一个MirroredStrategy的实例。这将处理同步训练的分布式策略。
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            # 超参数网格搜索
            params_grid_gcn_lstm = [
                {"in_feat": 1, "out_feat": out_feat, "lstm_units": lstm_units, "input_seq_len": input_seq_len,
                "forecast_horizon": 1, "multi_horizon": False, "batch_size": batch_size, "epochs": epochs,
                "learning_rate": learning_rate, "graph_conv_params": {"aggregation_type": "mean", 
                "combination_type": "concat", "activation": None}}
                for out_feat in [32, 64, 128]
                for lstm_units in [32, 64, 128]
                for batch_size in [32, 64, 128]
                for epochs in [20, 30, 40]
                for learning_rate in [0.001, 0.0001]
                for input_seq_len in [7, 14]
            ]

            params_grid_lstm = [
                {"lstm_units": lstm_units, "input_seq_len": input_seq_len, "batch_size": batch_size, 
                "epochs": epochs, "learning_rate": learning_rate}
                for lstm_units in [32, 64, 128]
                for batch_size in [32, 64, 128]
                for epochs in [20, 30, 40]
                for learning_rate in [0.001, 0.0001]
                for input_seq_len in [7, 14]
            ]

            # 运行网格搜索
            # 运行GCN+LSTM模型的网格搜索
            best_model_gcn_lstm, best_params_gcn_lstm, best_history_gcn_lstm, errors_gcn_lstm = grid_search_gcn_lstm(
                params_grid_gcn_lstm, num_nodes, graph, train_array
            )

            # 运行纯LSTM模型的网格搜索
            best_model_lstm, best_params_lstm, best_history_lstm, errors_lstm = grid_search_lstm(
                params_grid_lstm, train_array
            )

            # 保存GCN+LSTM模型的最佳参数
            best_params_gcn_lstm_list = pd.DataFrame(best_params_gcn_lstm)
            best_params_gcn_lstm_list.to_csv(os.path.join(save_path, f'best_params_gcn_lstm_{index}.csv'))

            # 保存纯LSTM模型的最佳参数
            best_params_lstm_list = pd.DataFrame(best_params_lstm)
            best_params_lstm_list.to_csv(os.path.join(save_path, f'best_params_lstm_{index}.csv'))

        # based on the best params, create test_dataset
        test_dataset = create_tf_dataset(test_array, best_params_gcn_lstm["input_seq_len"], 1, best_params_gcn_lstm["batch_size"], multi_horizon=False)
        test_dataset_lstm = create_tf_dataset_for_lstm(test_array, best_params_lstm["input_seq_len"], 1, test_array.shape[0])

        # 评估GCN+LSTM模型
        # save mae, mse, rmse, mmape to list
        gcn_lstm_mae_list[index], gcn_lstm_mse_list[index], gcn_lstm_rmse_list[index], gcn_lstm_mmape_list[index] = evaluate_model(
            best_model_gcn_lstm, test_dataset, best_history_gcn_lstm, save_path, index
        )
        
        # 评估纯LSTM模型
        # save mae, mse, rmse, mmape to list
        lstm_mae_list[index], lstm_mse_list[index], lstm_rmse_list[index], lstm_mmape_list[index] = evaluate_model_lstm(
            best_model_lstm, test_dataset_lstm, best_history_lstm, save_path, index
        )
    # save gcn+lstm's and lstm's mae, mse, rmse, mmape to csv, just need two csv, each have 4 columns
    # gcn+lstm's
    df = pd.DataFrame({'mae': gcn_lstm_mae_list, 'mse': gcn_lstm_mse_list, 'rmse': gcn_lstm_rmse_list, 'mmape': gcn_lstm_mmape_list})
    df.to_csv(os.path.join(save_path, f'GCN+LSTM_edged_weighted_global_Cluster_{cluster_label}.csv'))
    # lstm's
    df = pd.DataFrame({'mae': lstm_mae_list, 'mse': lstm_mse_list, 'rmse': lstm_rmse_list, 'mmape': lstm_mmape_list})
    df.to_csv(os.path.join(save_path, f'LSTM_edged_weighted_global_Cluster_{cluster_label}.csv'))
