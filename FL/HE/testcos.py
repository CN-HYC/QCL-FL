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

# 示例用法
if __name__ == "__main__":
    #秘钥分发，10个用户
    keyall = key_distribution(10)
    N = keyall["broadcast_pk"][0]
    pk = keyall["broadcast_pk"]
    sk1 = keyall["S1_keys"]["sk1"]
    sk2 = keyall["S2_keys"]["sk2"]
    nn = 10000
    #产生一个用户的正则化向量和一个用户的非正则化向量
    use1_v = generate_kd_vector(nn,normalize=True)
    use2_v = generate_kd_vector(nn)
    #两个用户向量加密
    jiami1 = time.time()
    use1_ev = [enc_pk(pk,Appr(x))["ciphertext"] for x in use1_v]
    use2_ev = [enc_pk(pk,Appr(x))["ciphertext"] for x in use2_v]
    jiami2 = time.time()
    start = time.time()
    #S1操作
    s1_r_use1 = generate_zn_star_random(N,nn)
    s1_r_use2 = generate_zn_star_random(N,nn)
    s1_er_use1 = [enc_pk(pk,x)["ciphertext"] for x in s1_r_use1]
    s1_er_use2 = [enc_pk(pk,x)["ciphertext"] for x in s1_r_use2]
    use1_aveev = [Mul(use1_ev[i],s1_er_use1[i],N) for i in range(nn)]
    use2_aveev = [Mul(use2_ev[i],s1_er_use2[i],N) for i in range(nn)]
    s1_use1_paveev = [part_dec(pk,x,sk1) for x in use1_aveev]
    s1_use2_paveev = [part_dec(pk,x,sk1) for x in use2_aveev]
    # S2操作
    s2_use1_paveev = [part_dec(pk,x,sk2) for x in use1_aveev]
    s2_use2_paveev = [part_dec(pk,x,sk2) for x in use2_aveev]
    s2_use1_faveev = [full_dec(pk,s1_use1_paveev[i],s2_use1_paveev[i]) for i in range(nn)]
    s2_use2_faveev = [full_dec(pk,s1_use2_paveev[i],s2_use2_paveev[i]) for i in range(nn)]
    s2_u1_sum_r = sum([(x * x) % (N * N) for x in s2_use1_faveev]) % (N*N)
    s2_u2_sum_r = sum([(x * x) % (N * N) for x in s2_use2_faveev]) % (N*N)
    s2_u1_esum_r = enc_pk(pk,s2_u1_sum_r%N)["ciphertext"]
    s2_u2_esum_r = enc_pk(pk,s2_u2_sum_r%N)["ciphertext"]
    #S1操作
    s1_u1_e2rk = [pow(x, 2 * r, N * N) for x, r in zip(use1_ev, s1_r_use1)]
    s1_u2_e2rk = [pow(x, 2 * r, N * N) for x, r in zip(use2_ev, s1_r_use2)]
    s1_u1_sr2 = sum([(r * r)%(N*N) for r in s1_r_use1])%(N*N)
    s1_u2_sr2 = sum([(r * r)%(N*N) for r in s1_r_use2])%(N*N)
    s1_u1_esr2 = enc_pk(pk, s1_u1_sr2%N)["ciphertext"]
    s1_u2_esr2 = enc_pk(pk, s1_u2_sr2%N)["ciphertext"]
    s1_u1_lcjg = 1
    s1_u2_lcjg = 1
    for i in range(nn):
        s1_u1_lcjg = (s1_u1_lcjg * s1_u1_e2rk[i])%(N*N)
    s1_u1_lcjg = (s1_u1_lcjg * s1_u1_esr2)%(N*N)
    for i in range(nn):
        s1_u2_lcjg = (s1_u2_lcjg * s1_u2_e2rk[i])%(N*N)
    s1_u2_lcjg = (s1_u2_lcjg * s1_u2_esr2)%(N*N)
    s1_u1_esum = (pow(s1_u1_lcjg, N - 1, N * N) * s2_u1_esum_r) % (N * N)
    s1_u2_esum = (pow(s1_u2_lcjg, N - 1, N * N) * s2_u2_esum_r) % (N * N)
    #S2操作
    s2_u1_psum = part_dec(pk, s1_u1_esum, sk2)
    s2_u2_psum = part_dec(pk, s1_u2_esum, sk2)
    #S1操作
    s1_u1_psum = part_dec(pk, s1_u1_esum, sk1)
    s1_u2_psum = part_dec(pk, s1_u2_esum, sk1)
    u1_sum = full_dec(pk, s2_u1_psum, s1_u1_psum)
    u2_sum = full_dec(pk, s2_u2_psum, s1_u2_psum)
    print(u1_sum/1e10-1<0.1)
    print(u2_sum/1e10-1<0.1)
    end = time.time()
    print(f'单次加密耗时: {(jiami2 - jiami1)/ 120} min')
    print(f'单次加密耗时: {(jiami2 - jiami1) / 2} s')
    print(f"单次服务器耗时: {(end - start)/120:.4f} min")
    print(f"单次服务器耗时: {(end - start)/2:.4f} s")








