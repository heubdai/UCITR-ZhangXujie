import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

data = np.resize(np.random.random(100),[2,50])
mask1 = [8,12,15,23,26,28,30,33]
# mask1 = [1,2,6,7,8,11,14,16,17,21,26,27,29,30,32,36,37,38,40,42,45,47,51,52,53,54,55,56,58,60,62,63,64,66,67,68,70,72,73,75,76,77,78,80,81,82,85,87,90]
mask2 = [8,12,23,28,33]
# mask2 = [1,5,23,28,33]
mask1 = np.array(mask1)
mask2 = np.array(mask2)
data1 = np.random.random(mask1.shape[0])
data2 = np.random.random(mask2.shape[0])
# print(data)
# print(mask1)
plt.figure()
# plt.subplot(2,1,1)
plt.plot(mask1, data1,marker='o')
plt.xticks([])  # 去掉横坐标值
plt.yticks([])  # 去掉纵坐标值
# plt.subplot(2,1,2)
plt.plot(mask2, data2,marker='o')
plt.xticks([])  # 去掉横坐标值
plt.yticks([])  # 去掉纵坐标值
plt.show()
# plt.savefig("test.jpg")
time_delat = mask1
for i in reversed(range(1,mask1.shape[0])):
    print(i)
    time_delat[i] = mask1[i]-mask1[i-1]
print(time_delat)