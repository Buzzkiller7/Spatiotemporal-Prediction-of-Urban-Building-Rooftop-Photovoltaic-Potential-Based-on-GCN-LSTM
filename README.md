# Spatiotemporal Prediction of Urban Building Rooftop Photovoltaic Potential Based on GCN-LSTM: A Case Study of Glasgow

**Chen Yang, Shengyuan Li, Zhonghua Gou\***
_School of Urban Design, Wuhan University, Wuhan, China;_
_\*correspondence: zh.gou@whu.edu.cn_

## Introduction

This repository contains the implementation of the study "Spatiotemporal Prediction of Urban Building Rooftop Photovoltaic Potential Based on GCN-LSTM: A Case Study of Glasgow." The study aims to predict the photovoltaic (PV) potential of urban building rooftops considering spatial shading relationships using a Graph Convolutional Network - Long Short-Term Memory (GCN-LSTM) model.

## Abstract

To address the building decarbonization crisis, the widespread adoption of rooftop photovoltaics (PV) has been agreed upon globally, with PV potential prediction being a crucial evaluation task. However, existing studies have deficiencies, particularly in urban-scale research, where the spatial shading relationships between buildings are often overlooked, resulting in predicted PV potentials that far exceed actual power generation. This study employs the GCN-LSTM model to perform spatiotemporal predictions of urban rooftop PV potential, taking into account spatial shading relationships between buildings. The results show that, compared to traditional Long Short-Term Memory (LSTM) models, GCN-LSTM significantly improves prediction accuracy, reducing MAE by 21%, MSE by 22%, RMSE by 13%, and MAPE by 12%. This improvement is particularly evident in winter and summer, validating the interpretability of the GCN-LSTM model.

## Keywords

- Photovoltaic potential
- Spatiotemporal prediction
- Spatial shading
- GCN-LSTM
- Graph robustness

![Graphical Abstract](./images/Graphical_Abstract.png)

## Installation

1. Clone the repository:
   ```sh
   git clone
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### **Code Description and Usage**

This repository provides several Python scripts that implement different components of the methodology. Below is the order of execution and a brief description of each script:

#### **1. `TopSolarRadiation_DNI_SIS.py`**

This script calculates the total solar radiation on sloped rooftops based on the Direct Normal Irradiance (DNI) and Surface Incoming Shortwave radiation (SIS) data. The calculated values are used as input for predicting rooftop PV potential.

**Usage:**

```
python scripts/TopSolarRadiation_DNI_SIS.py
```

#### **2. `SlantedListOpt_4Seasons.py`**

This script computes the shading relationships between buildings based on seasonal variations. It considers the effect of seasonal solar radiation on rooftop shading and calculates the shading effects for each building.

**Usage:**

```
python scripts/SlantedListOpt_4Seasons.py
```

#### **3. `Subgraph_Segmentation_Final.py`**

This script processes and segments the building graph into subgraphs. This segmentation is used to optimize the efficiency of the GCN-LSTM model during training.

**Usage:**

```
python scripts/Subgraph_Segmentation_Final.py
```

#### **4. `GCN_LSTM_GPU_GridSearch_GlobalCluster_EdgeWeight.py`**

This script performs global GCN-LSTM model training of building graphs based on spatial relationships.

**Usage:**

```
python scripts/GCN_LSTM_GPU_GridSearch_GlobalCluster_EdgeWeight.py
```

#### **5. `GCN_LSTM_GPU_GridSearch_4SeasonTrain.py`**

This script performs grid search for hyperparameter tuning of the GCN-LSTM model. It trains the model using data segmented by seasons (Spring, Summer, Autumn, and Winter) to find optimal parameters.

**Usage:**

```
python scripts/GCN_LSTM_GPU_GridSearch_4SeasonTrain.py
```

------

### **Model Training**

- **Data Preprocessing:** Before running the scripts, ensure you have the dataset prepared as described in the methodology of your paper (building morphological characteristics and solar radiation data).
- **Training:** The scripts sequentially process the data and perform grid search, clustering, and segmentation to train the GCN-LSTM model for each season's solar radiation prediction.

### **Running the Full Pipeline:**

1. **Step 1:** Run `TopSolarRadiation_DNI_SIS.py` to compute solar radiation data for rooftops.
2. **Step 2:** Execute `SlantedListOpt_4Seasons.py` to calculate building shading relationships considering seasonal variations.
3. **Step 3:** Use `Subgraph_Segmentation_Final.py` to finalize segmentation of building graphs.
4. **Step 4:** Run `GCN_LSTM_GPU_GridSearch_GlobalCluster_EdgeWeight.py` to training the global model.
5. **Step 5:** Finally, execute `GCN_LSTM_GPU_GridSearch_4SeasonTrain.py` for hyperparameter tuning and training the model across all four seasons.

## Repository Structure

- `data/`: Folder containing the dataset.
- `models/`: Folder containing the model implementations.
- `scripts/`: Folder containing preprocessing, training, and evaluation scripts.
- `results/`: Folder to store the results of the experiments.
- `README.md`: Project description and instructions.

## Contact

For any questions or suggestions, please contact 1285758029@qq.com.

## License

See the [LICENSE](LICENSE) file for details.
