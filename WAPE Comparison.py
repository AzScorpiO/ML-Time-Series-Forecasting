import numpy as np
import matplotlib.pyplot as plt

# Function to extract WAPE value from file
def get_wape_value(file_name):
    with open(file_name, 'r') as f:
        content = f.read().strip()
        return float(content)

# 读取文件中的WAPE值
cnn_64_wape = get_wape_value('cnn_64_wape.txt')
cnn_128_wape = get_wape_value('cnn_128_wape.txt')
lstm_50_wape = get_wape_value('lstm_50_wape.txt')
lstm_100_wape = get_wape_value('lstm_100_wape.txt')
mlp_100_wape = get_wape_value('mlp_100_wape.txt')
mlp_200_wape = get_wape_value('mlp_200_wape.txt')
rnn_50_wape = get_wape_value('rnn_50_wape.txt')
rnn_100_wape = get_wape_value('rnn_100_wape.txt')

# 文件名作为横坐标，WAPE值作为纵坐标
labels = ['CNN_64','CNN_128', 'LSTM_50', 'LSTM_100','MLP_100', 'MLP_200','RNN_50','RNN_100']
wape_values = [cnn_64_wape,cnn_128_wape, lstm_50_wape, lstm_100_wape, mlp_100_wape, mlp_200_wape,rnn_50_wape,rnn_100_wape]

# 创建图表
plt.figure(figsize=(16, 6))
plt.bar(labels, wape_values, color=['xkcd:blue','xkcd:orange','xkcd:red','xkcd:green','xkcd:pink','xkcd:teal','xkcd:violet','xkcd:cyan'])
plt.xlabel('Model')
plt.ylabel('WAPE Value')
plt.title('WAPE Comparison between Models')
plt.show()