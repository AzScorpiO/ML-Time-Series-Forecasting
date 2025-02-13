import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import time

# 记录程序开始时间
start_time = time.time()

# 加载数据集
data = pd.read_csv('electricity.csv')

# 数据预处理
# 提取用户用电量数据（假设从第3列到最后一列为用户用电量数据）
values = data.iloc[:, 2:].values

# 归一化数据
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# 创建时间序列数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, :])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_values, time_step)

# 调整输入数据形状以适应LSTM
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# 创建LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(X.shape[2]))

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, Y, epochs=50, batch_size=32, verbose=1)

# 进行预测
predictions = model.predict(X)

# 反归一化预测结果
predictions = scaler.inverse_transform(predictions)

# 输出预测结果
print(predictions)

# 记录程序结束时间并计算运行时间
end_time = time.time()
lstm_100_execution_time = end_time - start_time
print(f"{lstm_100_execution_time}")

# 将lstm_100_execution_time输出到lstm_100_execution_time.txt文件
with open('lstm_100_execution_time.txt', 'w') as f:
    f.write(f"{lstm_100_execution_time}")

# 计算WAPE值
def calculate_wape(actual, predicted):
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual))

# 反归一化实际值
actual_values = scaler.inverse_transform(scaled_values[time_step:time_step + len(predictions)])

wape_value = calculate_wape(actual_values[:len(predictions)], predictions)

# 将wape_value输出到lstm_100_wape.txt文件
with open('lstm_100_wape.txt', 'w') as f:
    f.write(f"{wape_value}")