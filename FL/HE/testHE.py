import math
import sympy
import random
from phe import paillier
import numpy as np
from Homomorphic import *
# 测试代码
def test_full_dec_workflow():
    """测试 FullDec 函数的端到端正确性"""
    try:
        print("=== 开始 FullDec 函数测试 ===")
        # 步骤1：生成密钥
        keys = key_gen(security_param_bits=1024)
        article_pk = keys["article_pk"]
        article_sk = keys["article_sk"]
        N = article_pk[0]
        print(f"1. 生成密钥：N={N}（{N.bit_length()}位），λ={article_sk}")

        # 步骤2：生成测试明文并加密
        test_x = 1234  # 任意测试明文
        enc_result = enc_pk(article_pk, test_x)
        ciphertext = enc_result["ciphertext"]
        print(f"2. 加密完成：明文x={test_x}，密文【x】={ciphertext}")

        # 步骤3：密钥拆分（生成 sk₁、sk₂）
        sk1, sk2 = key_split(article_sk, N)
        print(f"3. 密钥拆分：sk₁={sk1}，sk₂={sk2}")

        # 步骤4：执行部分解密（获取 [x]₁、[x]₂）
        partial1 = part_dec(article_pk, ciphertext, sk1)
        partial2 = part_dec(article_pk, ciphertext, sk2)
        print(f"4. 部分解密：[x]₁={partial1}[x]₂={partial2}")

        # 步骤5：执行完全解密（调用 FullDec 函数）
        decrypted_x = full_dec(article_pk, partial1, partial2)
        print(f"5. 完全解密结果：x={decrypted_x}")

        # 验证：解密结果与原明文是否一致
        print(f"\n=== 验证结果 ===")
        if decrypted_x == test_x:
            print(f"✅ FullDec 函数验证通过！解密结果与原明文一致（{decrypted_x} = {test_x}）")
        else:
            print(f"❌ FullDec 函数验证失败！解密结果（{decrypted_x}）与原明文（{test_x}）不一致")

    except Exception as e:
        print(f"\n❌ 测试过程出错：{str(e)}")


# 执行测试
if __name__ == "__main__":
    test_full_dec_workflow()