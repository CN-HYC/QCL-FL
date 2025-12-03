import numpy as np
from Homomorphic import *
from twotrapdoorHE import *
import time
from SecCos import *
if __name__ == "__main__":
    #秘钥分发，10个用户
    deg = 1e5
    n=10
    keyall = key_distribution(n)
    N = keyall["broadcast_pk"][0]
    pk = keyall["broadcast_pk"]
    sk1 = keyall["S1_keys"]["sk1"]
    sk2 = keyall["S2_keys"]["sk2"]
    nn = 100
    #产生n个随机梯度,以第一个梯度为基准线
    g = [generate_kd_vector(nn,normalize=True) for i in range(n)]
    jiami1 = time.time()
    eg = [[] for _ in range(n)]
    for i in range(n):
        eg[i] = [enc_pk(pk, Appr(x))["ciphertext"] for x in g[i]]
    jiami2 = time.time()
    start = time.time()
    cos = [[] for _ in range(n)]
    ncos = [[] for _ in range(n)]
    nratio = [[] for _ in range(n)]
    sumncos = 0
    for i in range(n):
        cos[i] = seccos(keyall,eg[0],eg[i])/deg
        ncos[i] = deg-cos[i]
        sumncos += ncos[i]
    for i in range(n):
        nratio[i] = Appr(ncos[i]/sumncos)
    egg = [pow(eg[0][i],nratio[0],N*N) for i in range(nn)]
    for i in range(1,n):
        for j in range(nn):
            egg[j]=(egg[j]*pow(eg[i][j],nratio[i],N*N))%(N*N)
    end = time.time()
    jiemi1 = time.time()
    agg = [part_dec(pk,egg[i],sk1) for i in range(nn)]
    bgg = [part_dec(pk,egg[i],sk2) for i in range(nn)]
    ggg = [full_dec(pk,agg[i],bgg[i]) for i in range(nn)]
    jiemi2 = time.time()
    gg = [0 for i in range(nn)]
    for i in range(n):
        for j in range(nn):
            gg[j]+=g[i][j]*(nratio[i]/1e5)*1e10
    print(ggg)
    print(gg)
    print(f'加密耗时: {(jiami2 - jiami1) / 60:.4f} min')
    print(f'加密耗时: {(jiami2 - jiami1):.4f} s')
    print(f"耗时: {(end - start) / 60:.4f} min")
    print(f"耗时: {(end - start)*1:.4f} s")
    print(f'解密耗时: {(jiemi2 - jiemi1) / 60:.4f} min')
    print(f'解密耗时: {(jiemi2 - jiemi1):.4f} s')