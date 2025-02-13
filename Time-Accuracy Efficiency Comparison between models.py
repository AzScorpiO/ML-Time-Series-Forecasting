import matplotlib.pyplot as plt

# 文件名列表
execution_time_files = [
    'cnn_64_execution_time.txt', 'cnn_128_execution_time.txt',
    'lstm_50_execution_time.txt', 'lstm_100_execution_time.txt',
    'mlp_100_execution_time.txt', 'mlp_200_execution_time.txt',
    'rnn_50_execution_time.txt', 'rnn_100_execution_time.txt'
]

wape_files = [
    'cnn_64_wape.txt', 'cnn_128_wape.txt',
    'lstm_50_wape.txt', 'lstm_100_wape.txt',
    'mlp_100_wape.txt', 'mlp_200_wape.txt',
    'rnn_50_wape.txt', 'rnn_100_wape.txt'
]

# 模型名称列表
model_names = [
    'CNN_64', 'CNN_128',
    'LSTM_50', 'LSTM_100',
    'MLP_100', 'MLP_200',
    'RNN_50', 'RNN_100'
]

# 读取文件内容
execution_times = []
wape_values = []

for file in execution_time_files:
    with open(file, 'r') as f:
        execution_times.append(float(f.read().strip()))

for file in wape_files:
    with open(file, 'r') as f:
        wape_values.append(float(f.read().strip()))

# 计算准确率（1 - WAPE）
accuracies = [1 - wape for wape in wape_values]

# 计算准确率除以时间的结果
accuracy_per_time = [acc / time for acc, time in zip(accuracies, execution_times)]

# 绘制柱状图
plt.figure(figsize=(12, 6))
plt.bar(model_names, accuracy_per_time, color=['xkcd:blue','xkcd:orange','xkcd:red','xkcd:green','xkcd:pink','xkcd:teal','xkcd:violet','xkcd:cyan'])
plt.xlabel('Model Names')
plt.ylabel('Accuracy / Time')
plt.title('Accuracy per Time for Different Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()