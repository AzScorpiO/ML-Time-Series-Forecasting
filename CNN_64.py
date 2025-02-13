import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
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

# 调整输入数据形状以适应Conv1D
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# 创建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, X.shape[2])))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(X.shape[2]))

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, Y, epochs=50, batch_size=32, verbose=1)

# 进行预测
predictions = model.predict(X)

# 反归一化预测结果
predictions = scaler.inverse_transform(predictions)

print(predictions)

# 记录程序结束时间并计算运行时间
end_time = time.time()
cnn_64_execution_time = end_time - start_time
print(f"CNN运行时间为: {cnn_64_execution_time} 秒")

# 将cnn_execution_time输出到cnn_execution_time.txt文件
with open('cnn_64_execution_time.txt', 'w') as f:
    f.write(f"{cnn_64_execution_time}")

# 计算WAPE值
def calculate_wape(actual, predicted):
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual))

# 反归一化实际值
actual_values = scaler.inverse_transform(scaled_values[time_step:time_step + len(predictions)])

wape_value = calculate_wape(actual_values[:len(predictions)], predictions)

# 将wape_value输出到cnn_wape.txt文件
with open('cnn_64_wape.txt', 'w') as f:
    f.write(f"{wape_value}")
