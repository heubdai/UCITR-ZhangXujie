import numpy as np
import random

SIGUA = ['老阴','少阳','少阴','老阳']
BAGUA = ['坤','震','坎','兑','艮','离','巽','乾']
BAGUA_name = ['地','雷','水','泽','山','火','风','天']
LIUSHISIGUA = []

def get_gua():
    gua = np.zeros(6)
    for i in range(6):
        total = 3
        for j in range(3):
            s = random.randint(0, 1)
            total -= s
        gua[i] = total
# 得到六个数，依据sigua对应老阴到老阳
    return gua

def get_bagua(gua):
    ba = np.zeros(6)
    for i in range(6):
        if gua[i]%2==0:
            ba[i] = 0
        else:
            ba[i]=1
    bagua = np.zeros(2)
    bagua[0] = np.sum(ba[0]+ba[1]*2+ba[2]*4)
    bagua[1] = np.sum(ba[3]+ba[4]*2+ba[5]*4)
    return bagua
    # 返回解析的两个八卦，前一个为内卦，在下

if __name__ == "__main__":
    gua = get_gua()
    print(gua)
    bagua = get_bagua(gua)
    print(bagua)