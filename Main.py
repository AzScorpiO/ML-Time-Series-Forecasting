
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from CNN import train_and_predict_cnn
from LSTM import train_and_predict_lstm
from RNN import train_and_predict_rnn
from MLP import train_and_predict_mlp

# 加载数据集
data = pd.read_csv('electricity.csv')

# 数据预处理
values = data.iloc[:, 2:].values
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, :])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_values, time_step)

# 定义WAPE指标
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# 训练和预测模型
cnn_predictions = train_and_predict_cnn(X, Y, scaler)
lstm_predictions = train_and_predict_lstm(X, Y, scaler)
rnn_predictions = train_and_predict_rnn(X, Y, scaler)
mlp_predictions = train_and_predict_mlp(X.reshape(X.shape[0], X.shape[1] * X.shape[2]), Y.reshape(Y.shape[0], Y.shape[1]), scaler)

# 计算WAPE
cnn_wape = wape(values[time_step:], cnn_predictions)
lstm_wape = wape(values[time_step:], lstm_predictions)
rnn_wape = wape(values[time_step:], rnn_predictions)
mlp_wape = wape(values[time_step:], mlp_predictions)

# 输出WAPE
print(f'CNN WAPE: {cnn_wape}')
print(f'LSTM WAPE: {lstm_wape}')
print(f'RNN WAPE: {rnn_wape}')
print(f'MLP WAPE: {mlp_wape}')

# 可视化比较结果
models = ['CNN', 'LSTM', 'RNN', 'MLP']
wape_values = [cnn_wape, lstm_wape, rnn_wape, mlp_wape]

plt.figure(figsize=(10, 6))
plt.bar(models, wape_values, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Comparison using WAPE')
plt.xlabel('Models')
plt.ylabel('WAPE')
plt.show()
