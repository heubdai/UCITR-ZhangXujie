import csv

import numpy as np

with open("../datasets/electricity/electricity.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    # 逐行获取数据，并输出
    data = np.zeros((26304, 320))
    i=0
    for row in reader:
        data[i] = np.array(row[1:321])
        i+=1

print(data.shape)
print(data[-1])
    # print(reader)
