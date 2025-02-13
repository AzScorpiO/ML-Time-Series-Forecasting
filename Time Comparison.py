import matplotlib.pyplot as plt

# Function to extract execution time from file
def get_execution_time(file_name):
    with open(file_name, 'r') as f:
        content = f.read().strip()
        return float(content)

# 读取文件中的数字
try:
    cnn_64_time = get_execution_time('cnn_64_execution_time.txt')
    cnn_128_time = get_execution_time('cnn_128_execution_time.txt')
    lstm_50_time = get_execution_time('lstm_50_execution_time.txt')
    lstm_100_time = get_execution_time('lstm_100_execution_time.txt')
    mlp_100_time = get_execution_time('mlp_100_execution_time.txt')
    mlp_200_time = get_execution_time('mlp_200_execution_time.txt')
    rnn_50_time = get_execution_time('rnn_50_execution_time.txt')
    rnn_100_time = get_execution_time('rnn_100_execution_time.txt')
except FileNotFoundError as e:
    print(e)
    raise
except ValueError as e:
    print(e)
    raise

# 文件名作为横坐标，时间作为纵坐标
labels = ['CNN_64','CNN_128', 'LSTM_50', 'LSTM_100','MLP_100', 'MLP_200','RNN_50','RNN_100']
times = [cnn_64_time,cnn_128_time, lstm_50_time, lstm_100_time, mlp_100_time, mlp_200_time,rnn_50_time,rnn_100_time]

# 创建图表
plt.figure(figsize=(16, 6))
plt.bar(labels, times, color=['xkcd:blue','xkcd:orange','xkcd:red','xkcd:green','xkcd:pink','xkcd:teal','xkcd:violet','xkcd:cyan'])
plt.xlabel('Model')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.show()