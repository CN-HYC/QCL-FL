import math
import sympy
import random
from phe import paillier  # 导入phe库

def key_gen(security_param_bits=1024):
    """
    实现文章中的KeyGen(ε)→(pk, sk)密钥生成函数
    兼容无generate_private_key方法的早期phe版本，使用标准密钥生成函数
    :param security_param_bits: 安全参数（N的位数），默认1024位
    :return: 包含文章定义的公钥、私钥及phe库兼容密钥的字典
    """
    # 1. 生成两个不同的奇素数p和q（符合文章"distinct odd primes"要求）
    prime_length = security_param_bits // 2
    while True:
        p = sympy.randprime(2 **(prime_length - 1), 2** prime_length - 1)
        q = sympy.randprime(2 **(prime_length - 1), 2** prime_length - 1)
        if p != q:
            break

    # 2. 计算文章定义的核心参数
    N = p * q  # 大整数N = p*q
    article_pk = (N, 1 + N)  # 文章定义的公钥 (N, 1+N)
    # 文章定义的私钥 sk = λ = lcm(p-1, q-1)
    gcd_val = math.gcd(p - 1, q - 1)
    article_sk = (p - 1) * (q - 1) // gcd_val

    # 3. 兼容早期phe版本：用标准函数生成密钥对（不依赖generate_private_key）
    # 注意：phe的generate_paillier_keypair会自动生成p和q，但我们需要强制使用自己生成的p和q
    # 因此这里通过私钥构造函数手动传入p和q（兼容所有版本的底层逻辑）
    phe_public_key = paillier.PaillierPublicKey(n=N)
    # 直接实例化私钥（早期版本兼容写法）
    phe_private_key = paillier.PaillierPrivateKey(phe_public_key, p, q)

    return {
        "article_pk": article_pk,          # 文章定义的公钥
        "article_sk": article_sk,          # 文章定义的私钥λ
        "phe_private_key": phe_private_key,  # phe库私钥
        "phe_public_key": phe_public_key    # phe库公钥（g = N+1）
    }

def enc_pk(article_pk, x):
    """
    文章定义的加密函数：Enc_pk(x) → 【x】
    :param article_pk: 文章定义的公钥，格式为(N, 1+N)（从key_gen获取）
    :param x: 明文，需满足x ∈ Z_N（0 ≤ x < N）
    :return: 密文【x】，即(1+N)^x * r^N mod N²
    """
    # 从公钥中提取N（核心参数）
    N = article_pk[0]
    one_plus_N = article_pk[1]  # 1+N（公钥的第二个元素）
    N_square = N * N  # 计算N²（模运算的基数）

    # 1. 校验并处理明文x：确保x ∈ Z_N（若超出范围，对N取模）
    if not (0 <= x < N):
        x = x % N
        print(f"明文x超出Z_N范围，已自动取模N处理，处理后x={x}")

    # 2. 生成随机数r ∈ Z_N*（r与N互质，即gcd(r, N) == 1）
    while True:
        r = random.randint(1, N - 1) # 生成1~N-1的随机整数
        if math.gcd(r, N) == 1:  # 验证r与N互质
            break

    # 3. 按公式计算密文：(1+N)^x * r^N mod N²
    # 用pow(a, b, mod)优化大数计算，避免溢出
    term1 = pow(one_plus_N, x, N_square)  # (1+N)^x mod N²
    term2 = pow(r, N, N_square)  # r^N mod N²
    ciphertext = (term1 * term2) % N_square  # 最终密文：(term1*term2) mod N²

    return {
        "ciphertext": ciphertext,  # 文章定义的密文【x】
        "r": r  # 生成的随机数r（用于调试/验证）
    }


def key_split(sk_lambda, N):
    """
    文章定义的密钥拆分函数：KeySplit(sk)→(sk₁,sk₂)
    满足两个同余条件：
    1. sk₁ + sk₂ ≡ 0 mod λ (sk_lambda即λ)
    2. sk₁ + sk₂ ≡ 1 mod N
    :param sk_lambda: 文章中的sk=λ（从key_gen获取）
    :param N: 公钥中的核心参数N（从key_gen获取）
    :return: (sk₁, sk₂) - 满足条件的两个密钥份额
    """

    # 步骤1：推导sk₁+sk₂的取值（记S=sk₁+sk₂）
    # 由条件1：S ≡ 0 mod λ → S = m*λ（m为正整数）
    # 由条件2：m*λ ≡ 1 mod N → 需先求m（m是λ在模N下的逆元）
    # 因λ=lcm(p-1,q-1)，N=pq，gcd(λ,N)=1，逆元存在，用扩展欧几里得算法求逆
    def extended_gcd(a, b):
        """扩展欧几里得算法，求a在模b下的逆元"""
        if b == 0:
            return a, 1, 0
        gcd_val, x, y = extended_gcd(b, a % b)
        return gcd_val, y, x - (a // b) * y

    gcd_val, m, _ = extended_gcd(sk_lambda, N)
    if gcd_val != 1:
        raise ValueError("λ与N不互质，无法满足同余条件（理论上不应出现）")
    # 确保m为正（逆元可能为负，需调整为模N的正逆元）
    m = m % N
    S = m * sk_lambda  # S=sk₁+sk₂，同时满足S≡0 modλ和S≡1 modN

    # 步骤2：随机生成sk₁（范围1~S-1，确保sk₂=S-sk₁为正）
    sk1 = random.randint(1, S - 1)
    # 步骤3：计算sk₂=S-sk₁（自动满足S=sk₁+sk₂，即两个同余条件）
    sk2 = S - sk1

    # 验证条件（可选，用于调试确保正确性）
    assert (sk1 + sk2) % sk_lambda == 0, "未满足sk₁+sk₂≡0 modλ"
    assert (sk1 + sk2) % N == 1, "未满足sk₁+sk₂≡1 modN"

    return sk1, sk2

def part_dec(article_pk, ciphertext, sk_i):
    """
    文章定义的部分解密函数：PartDecₛₖᵢ(【x】)→[x]ᵢ
    公式：[x]ᵢ = 【x】^sk_i mod N²
    :param article_pk: 文章定义的公钥（含N，从key_gen获取）
    :param ciphertext: 加密后的密文【x】（从前文enc_pk函数获取）
    :param sk_i: 单个密钥份额（sk₁或sk₂，从key_split函数获取）
    :return: [x]ᵢ - 部分解密结果（用于后续FullDec）
    """
    N = article_pk[0]
    N_square = N * N  # 模运算基数N²（文章公式要求）

    # 用pow函数高效计算大数幂模（避免溢出，Python内置优化）
    partial_decrypt = pow(ciphertext, sk_i, N_square)
    return partial_decrypt


def full_dec(article_pk, partial1, partial2):
    """
    文章定义的完全解密函数：FullDec([x]₁, [x]₂)→x
    公式：x = [(∏ᵢ₌₁²[x]ᵢ mod N²) - 1] / N mod N
    :param article_pk: 文章定义的公钥（含N，从key_gen获取）
    :param partial1: 第一个部分解密结果 [x]₁（从part_dec获取）
    :param partial2: 第二个部分解密结果 [x]₂（从part_dec获取）
    :return: 明文 x（∈ Z_N）
    """
    # 1. 从公钥提取核心参数 N 和 N²
    N = article_pk[0]
    N_square = N * N  # 公式中 mod N² 的基数

    # 2. 计算两个部分解密结果的乘积，并对 N² 取模（∏ᵢ₌₁²[x]ᵢ mod N²）
    prod = (partial1 * partial2) % N_square

    # 3. 计算 (prod - 1) 并验证整除性（理论上 prod-1 必能被 N 整除，因Paillier加密性质）
    prod_minus_1 = prod - 1
    if prod_minus_1 % N != 0:
        raise ValueError("部分解密结果异常，(prod - 1) 无法被 N 整除（理论上不应发生）")

    # 4. 按公式计算明文 x：(prod_minus_1 / N) mod N
    x = (prod_minus_1 // N) % N  # 用整数除法//，避免浮点数误差

    return x

