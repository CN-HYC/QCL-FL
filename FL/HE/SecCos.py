import numpy as np
from Homomorphic import *
from twotrapdoorHE import *
import time
def key_distribution(n, security_param_bits=1024):
    """
    实现密钥分发流程
    :param n: 用户数量
    :param security_param_bits: 安全参数位数
    :return: 包含所有分发密钥的字典
    """
    # 步骤1: KC生成主密钥对 (pk, sk)
    key_info = key_gen(security_param_bits)
    pk = key_info["article_pk"]
    sk = key_info["article_sk"]
    N = key_info["article_pk"][0]

    # 步骤2: KC拆分sk得到(sk1, sk2)
    sk1, sk2 = key_split(sk, N)

    # 步骤3: 为每个用户生成(sk_ui, sk_si)
    user_keys = []
    for i in range(n):
        sk_ui, sk_si = key_split(sk, N)  # 每个用户独立拆分
        user_keys.append({
            "user_id": i + 1,
            "sk_ui": sk_ui,  # 用户Ui持有的密钥
            "sk_si": sk_si  # 对应S1持有的用户相关密钥
        })

    # 步骤4: 整理分发结果
    distribution_result = {
        "broadcast_pk": pk,  # 广播的公钥
        "S1_keys": {
            "sk1": sk1,  # S1的主密钥份额
            "user_sk_si": [uk["sk_si"] for uk in user_keys]  # 各用户对应的sk_si
        },
        "S2_keys": {
            "sk2": sk2  # S2的密钥份额
        },
        "user_keys": [{"user_id": uk["user_id"], "sk_ui": uk["sk_ui"]} for uk in user_keys]
    }

    return distribution_result

def generate_kd_vector(k, distribution='uniform', normalize=False, **kwargs):
    """
    生成k维随机向量，支持生成正则化（L2模长=1）的向量

    参数:
        k: 向量维度（正整数）
        distribution: 随机分布类型
            - 'uniform': 均匀分布（默认）
            - 'normal': 正态分布
            - 'integer': 整数分布
        normalize: 是否将向量正则化（L2模长=1），默认False
        **kwargs: 分布参数
            - 均匀分布: low=0.0, high=1.0（数值范围）
            - 正态分布: loc=0.0, scale=1.0（均值和标准差）
            - 整数分布: low=0, high=10（整数范围，左闭右开）

    返回:
        numpy数组: 长度为k的随机向量（若normalize=True则L2模长=1）
    """
    # 输入参数校验
    if not isinstance(k, int) or k <= 0:
        raise ValueError("维度k必须是正整数")

    # 1. 生成基础随机向量
    if distribution == 'uniform':
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 1.0)
        vec = np.random.uniform(low, high, size=k)

    elif distribution == 'normal':
        loc = kwargs.get('loc', 0.0)
        scale = kwargs.get('scale', 1.0)
        vec = np.random.normal(loc, scale, size=k)

    elif distribution == 'integer':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 10)
        vec = np.random.randint(low, high, size=k).astype(float)  # 整数转浮点便于后续正则化

    else:
        raise ValueError(f"不支持的分布类型: {distribution}")

    # 2. 若需要正则化（L2范数=1）
    if normalize:
        # 计算L2范数（模长）
        l2_norm = np.linalg.norm(vec)
        # 避免除以零（概率极低，主要针对均匀/整数分布可能出现的全零向量）
        if np.isclose(l2_norm, 0):
            raise RuntimeWarning("生成的向量接近零向量，无法正则化，返回原始向量")
        # 正则化：每个元素除以L2范数
        vec = vec / l2_norm

    return vec

def generate_zn_star_random(N, k):
    """
    生成k个属于Z_N*的随机噪声r（Z_N*：模N的乘法群，即r满足1≤r<N且gcd(r,N)=1）
    :param N: 公钥中的核心参数N（来自key_gen生成的article_pk[0]）
    :param k: 需生成的噪声数量（对应选中内容中的m）
    :return: 噪声列表[r₁, r₂, ..., rₖ]，每个r∈Z_N*
    """
    zn_star_noises = []
    while len(zn_star_noises) < k:
        # 生成1~N-1范围内的随机整数
        r = random.randint(1, N - 1)
        # 验证r与N互质（gcd(r,N)=1），满足Z_N*定义
        if math.gcd(r, N) == 1:
            zn_star_noises.append(r)
    return zn_star_noises

def cosine_similarity_np(vec_a, vec_b):
    """
    用NumPy计算两个向量的余弦相似度

    参数:
        vec_a: 第一个向量（numpy数组）
        vec_b: 第二个向量（numpy数组）

    返回:
        float: 余弦相似度（范围[-1, 1]）
    """
    # 计算点积
    dot_product = np.dot(vec_a, vec_b)

    # 计算模长
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # 避免除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def seccos(keyall,ega,egb):
    N = keyall["broadcast_pk"][0]
    pk = keyall["broadcast_pk"]
    sk1 = keyall["S1_keys"]["sk1"]
    sk2 = keyall["S2_keys"]["sk2"]
    nn = len(ega)
    #S1
    s1_r = generate_zn_star_random(N,nn)
    s1_er = [enc_pk(pk,x)["ciphertext"] for x in s1_r]
    ega_er = [Mul(ega[i],s1_er[i],N) for i in range(nn)]
    egb_er = [Mul(egb[i],s1_er[i],N) for i in range(nn)]
    p1ega_er = [part_dec(pk, x, sk1) for x in ega_er]
    p1egb_er = [part_dec(pk, x, sk1) for x in egb_er]
    #S2操作
    p2ega_er = [part_dec(pk, x, sk2) for x in ega_er]
    p2egb_er = [part_dec(pk, x, sk2) for x in egb_er]
    fega_er = [full_dec(pk, p1ega_er[i], p2ega_er[i]) for i in range(nn)]
    fegb_er = [full_dec(pk, p1egb_er[i], p2egb_er[i]) for i in range(nn)]
    zipab = zip(fega_er,fegb_er)
    cosab_er = sum([Mul(x,y,N) for x,y in zipab])%(N*N)
    ecosab_er = enc_pk(pk,cosab_er%N)["ciphertext"]
    #S1操作
    sr2 = sum([(r * r)%(N*N) for r in s1_r])%(N*N)
    esr2 = enc_pk(pk, sr2%N)["ciphertext"]
    arzip = zip(ega,s1_r)
    brzip = zip(egb,s1_r)
    ar = [pow(x,r,N*N) for x,r in arzip]
    br = [pow(x,r,N*N) for x,r in brzip]
    sar = 1
    sbr = 1
    for i in ar:
        sar = Mul(sar,i,N)
    for i in br:
        sbr = Mul(sbr,i,N)
    sz = (sar*sbr*esr2)%(N*N)
    ecos = (pow(sz, N - 1, N * N) * ecosab_er) % (N * N)
    #S2操作
    p2cos = part_dec(pk, ecos, sk2)
    #S1操作
    p1cos = part_dec(pk, ecos, sk1)
    cosab = full_dec(pk, p1cos, p2cos)
    return cosab

if __name__ == "__main__":
    #秘钥分发，10个用户
    keyall = key_distribution(10)
    N = keyall["broadcast_pk"][0]
    pk = keyall["broadcast_pk"]
    sk1 = keyall["S1_keys"]["sk1"]
    sk2 = keyall["S2_keys"]["sk2"]
    nn = 10000
    #产生两个随机梯度
    ga = generate_kd_vector(nn, normalize=True)
    gb = generate_kd_vector(nn, normalize=True)
    jiami1 = time.time()
    ega = [enc_pk(pk, Appr(x))["ciphertext"] for x in ga]
    egb = [enc_pk(pk, Appr(x))["ciphertext"] for x in gb]
    jiami2 = time.time()
    start = time.time()
    abcos = seccos(keyall,ega,egb)
    end = time.time()
    print()
    print(abcos, abcos / 1e5)
    print(cosine_similarity_np(ga,gb))
    print(f'单次加密耗时: {(jiami2 - jiami1) / 60:.4f} min')
    print(f'单次加密耗时: {(jiami2 - jiami1):.4f} s')
    print(f"耗时: {(end - start) / 60:.4f} min")
    print(f"耗时: {(end - start)*1:.4f} s")
