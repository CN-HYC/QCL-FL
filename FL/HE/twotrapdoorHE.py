import math
import sympy
import random
from phe import paillier  # 导入phe库
from Homomorphic import *

def Appr(x, deg=10 ** 5):
    """浮点数转整数：x' = round(x * deg)"""
    return round(x * deg)


import math


def HE_Add(article_pk, ciphertexts):
    """
    同态加法：计算多个密文的和的密文 [[a + b + ...]]
    参数：
        article_pk: 文章定义的公钥 (N, 1+N)
        ciphertexts: 密文列表 [ [a], [b], ... ]（每个元素为整数密文）
    返回：
        求和结果的密文 [[a + b + ...]]
    """
    if not ciphertexts:
        raise ValueError("密文列表不能为空")

    N = article_pk[0]
    N_square = N * N  # 模N²运算
    result = ciphertexts[0]

    # 同态加法：密文相乘等价于明文相加（模N²）
    for c in ciphertexts[1:]:
        result = (result * c) % N_square

    return result

def Mul(x,y,N):
    N_square = N * N  # 模N²运算
    return (x*y) % N_square