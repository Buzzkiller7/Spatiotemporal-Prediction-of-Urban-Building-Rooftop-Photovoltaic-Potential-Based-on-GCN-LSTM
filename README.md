# Spatiotemporal Prediction of Urban Building Rooftop Photovoltaic Potential Based on GCN-LSTM: A Case Study of Glasgow

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

## Usage

1. Prepare your dataset following the structure provided in the `data` folder.
2. Run the preprocessing script:
   ```sh
   python preprocess.py
   ```
3. Train the model:
   ```sh
   python train.py
   ```
4. Evaluate the model:
   ```sh
   python evaluate.py
   ```

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
