/*
 * Copyright (c) 2018 XLAB d.o.o.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sodium.h>
#include <amcl/pair_BN254.h>
#include <amcl/pbc_support.h>

#include "cifer/internal/big.h"
#include "cifer/innerprod/fullysec/dmcfe.h"
#include "cifer/internal/dlog.h"
#include "cifer/internal/hash.h"
#include "cifer/sample/uniform.h"

void cfe_dmcfe_client_init(cfe_dmcfe_client *c, size_t idx) {
    mpz_t client_sec_key; //私钥
    mpz_inits(c->order, client_sec_key, NULL);//初始化order和私钥sk

    c->idx = idx;//客户编号idx
    BIG_256_56_rcopy(c->order_big, CURVE_Order_BN254);
    mpz_from_BIG_256_56(c->order, c->order_big);
    cfe_vec_init(&(c->s), 2);
    cfe_uniform_sample_vec(&(c->s), c->order);//s为2维随机向量 大小为[0，order]

    ECP_BN254_generator(&(c->client_pub_key));
    cfe_uniform_sample(client_sec_key, c->order);
    BIG_256_56_from_mpz(c->client_sec_key_big, client_sec_key);//生成私钥client_sec_key_big  sk
    ECP_BN254_mul(&(c->client_pub_key), c->client_sec_key_big);//计算公钥client_pub_key （g^sk）

    cfe_mat_init(&(c->share), 2, 2);//share矩阵2x2
    mpz_clear(client_sec_key);
}

//释放客户端内存
void cfe_dmcfe_client_free(cfe_dmcfe_client *c) {
    mpz_clear(c->order);
    cfe_mat_free(&(c->share));
    cfe_vec_free(&(c->s));
}

//share即为密钥交换然后哈希得到的可消去掩码矩阵
void cfe_dmcfe_set_share(cfe_dmcfe_client *c, ECP_BN254 *pub_keys, size_t num_clients) {
    cfe_mat add;
    cfe_mat_init(&add, 2, 2);
    ECP_BN254 shared_g1;
    char h1[MODBYTES_256_56 + 1];
    octet tmp_oct = {0, sizeof(h1), h1};
    char h2[randombytes_SEEDBYTES];
    octet tmp_hash = {0, sizeof(h2), h2};
    for (size_t k = 0; k < num_clients; k++) {
        if (k == c->idx) {
            continue;
        }

        ECP_BN254_copy(&shared_g1, &(pub_keys[k]));//计算shared_g1 = pub_keys[k] * c->client_sec_key_big
        ECP_BN254_mul(&shared_g1, c->client_sec_key_big);
        ECP_BN254_toOctet(&tmp_oct, &shared_g1, true);//转换shared_g1为字节数组tmp_oct

        mhashit(SHA256, -1, &tmp_oct, &tmp_hash);//哈希tmp_oct得到tmp_hash

        cfe_uniform_sample_mat_det(&add, c->order, ((unsigned char *) tmp_hash.val));//用hash结果得到随机矩阵

        if (k > c->idx) {//k>c->inx 取负数
            cfe_mat_neg(&add, &add);
        }
        cfe_mat_add(&(c->share), &(c->share), &add);
        cfe_mat_mod(&(c->share), &(c->share), c->order);//将add加到c->share上取模
    }

    cfe_mat_free(&add);
}

void cfe_dmcfe_encrypt(ECP_BN254 *cipher, cfe_dmcfe_client *c, mpz_t x, char *label, size_t label_len) {
    ECP_BN254 h;
    BIG_256_56 tmp_big;
    ECP_BN254_inf(cipher);
    cfe_string label_str = {label, label_len};
    cfe_string space_str = {(char *) " ", 1};
    cfe_string i_str, label_for_hash;
    for (int i = 0; i < 2; i++) {//计算哈希标签H（0 label）和H（1 label）
        cfe_int_to_str(&i_str, i);
        cfe_strings_concat(&label_for_hash, &i_str, &space_str, &label_str, NULL);
        cfe_hash_G1(&h, &label_for_hash);//G1 Hash 标签
        BIG_256_56_from_mpz(tmp_big, c->s.vec[i]);//将客户的加密密钥s（二维中的1）转换为256类型tmp_big
        ECP_BN254_mul(&h, tmp_big);//哈希值H（0||label）乘以加密密钥s  h=tmp_big（s）*H（0||label）
        ECP_BN254_add(cipher, &h);//得到密文cipher=0+s1*h1
        cfe_string_free(&label_for_hash);
        cfe_string_free(&i_str);
    }

    BIG_256_56_from_mpz(tmp_big, x);//将明文x同样作类型转换tmp_big
    ECP_BN254_generator(&h);
    ECP_BN254_mul(&h, tmp_big);//x乘上生成元
    ECP_BN254_add(cipher, &h);//得到最终密文：s1*h1+s2*h2+g*x
}

void cfe_dmcfe_fe_key_part_init(cfe_vec_G2 *key_share) {
    cfe_vec_G2_init(key_share, 2);
}

void cfe_dmcfe_derive_fe_key_part(cfe_vec_G2 *fe_key_part, cfe_dmcfe_client *c, cfe_vec *y) {
    cfe_string str, str_i, for_hash;
    cfe_string space_str = {(char *) " ", 1};
    cfe_vec_to_string(&str, y);
    ECP2_BN254 hash[2];
    for (int i = 0; i < 2; i++) {
        cfe_int_to_str(&str_i, i);
        cfe_strings_concat(&for_hash, &str_i, &space_str, &str, NULL); 
        cfe_hash_G2(&(hash[i]), &for_hash);//G2 Hash[]标签(二维) 计算哈希标签H（0 y）和H（1 y）
        cfe_string_free(&for_hash);
        cfe_string_free(&str_i);
    }
    cfe_string_free(&str);

    mpz_t tmp;
    mpz_init(tmp);
    BIG_256_56 tmp_big;
    ECP2_BN254 h;
    for (size_t k = 0; k < 2; k++) {
        ECP2_BN254_inf(&(fe_key_part->vec[k]));//部分解密密钥vec[]初始化为空
        for (size_t i = 0; i < 2; i++) {
        ECP2_BN254_copy(&h, &(hash[i]));//将G2 hash[0 y]赋值给h
            cfe_mat_get(tmp, &(c->share), k, i);//取出共享矩阵share的元素给tmp (先第一行)
            BIG_256_56_from_mpz(tmp_big, tmp);
            ECP2_BN254_mul(&(h), tmp_big);//tmp与hash结果相乘
            ECP2_BN254_add(&(fe_key_part->vec[k]), &h);//累加到vec  vec[0]=share[0,0]*h+share[0,1]*h  vec[1]=share[1,0]*h+share[1,1]*h
        }

        mpz_mul(tmp, y->vec[c->idx], c->s.vec[k]);//y[i]与密钥s[0]相乘赋值给tmp，下一步mod，然后转换类型，乘以生成元h
        mpz_mod(tmp, tmp, c->order);
        BIG_256_56_from_mpz(tmp_big, tmp);
        ECP2_BN254_generator(&h);
        ECP2_BN254_mul(&h, tmp_big);
        ECP2_BN254_add(&(fe_key_part->vec[k]), &h);//部分解密密钥vec[0]=(share[0,0]*h+share[0,1]*h)*g^(yi*s0) mod p
    }                                                         //vec[1]=(share[1,0]*h+share[1,1]*h)*g^(yi*s1) mod p
    mpz_clear(tmp);
}

cfe_error cfe_dmcfe_decrypt(mpz_t res, ECP_BN254 *ciphers, cfe_vec_G2 *key_shares,
                            char *label, size_t label_len, cfe_vec *y, mpz_t bound) {
    cfe_vec_G2 keys_sum;
    cfe_vec_G2_init(&keys_sum, 2);
    for (size_t i = 0; i < 2; i++) {
        ECP2_BN254_inf(&(keys_sum.vec[i]));//初始化key_sum
    }

    // Ensure key_shares is properly initialized
    for (size_t k = 0; k < y->size; k++) {
        if (key_shares[k].size != 2) {
            fprintf(stderr, "Invalid size of key_shares[%zu]\n", k);
            return -1;
        }
        for (size_t i = 0; i < 2; i++) {
            ECP2_BN254_add(&(keys_sum.vec[i]), &(key_shares[k].vec[i]));//累加所有客户的vec[0]、vec[1]得到key_sum[0,1]
        }
    }

    ECP_BN254 ciphers_sum, cipher_i, gen1, h;
    ECP2_BN254 gen2;
    FP12_BN254 s, t, pair;
    ECP_BN254_generator(&gen1);
    ECP2_BN254_generator(&gen2);
    ECP_BN254_inf(&ciphers_sum);
    BIG_256_56 y_i;
    mpz_t y_i_mod, order;
    mpz_inits(y_i_mod, order, NULL);
    mpz_from_BIG_256_56(order, (int64_t *) CURVE_Order_BN254);

    for (size_t i = 0; i < y->size; i++) {
        ECP_BN254_copy(&cipher_i, &(ciphers[i]));
        mpz_mod(y_i_mod, y->vec[i], order);
        BIG_256_56_from_mpz(y_i, y_i_mod);
        ECP_BN254_mul(&cipher_i, y_i);
        ECP_BN254_add(&ciphers_sum, &cipher_i);//对应密文与yi相乘求和
    }

    PAIR_BN254_ate(&s, &gen2, &ciphers_sum);
    PAIR_BN254_fexp(&s);

    cfe_string label_for_hash, str_i;
    cfe_string label_str = {label, label_len};
    cfe_string space_str = {(char *) " ", 1};
    FP12_BN254_one(&t);
    for (int i = 0; i < 2; i++) {
        cfe_int_to_str(&str_i, i);
        cfe_strings_concat(&label_for_hash, &str_i, &space_str, &label_str, NULL);
        cfe_hash_G1(&h, &label_for_hash);
        PAIR_BN254_ate(&pair, &(keys_sum.vec[i]), &h);//配对e（sum.vec[1],h(1 label)）
        PAIR_BN254_fexp(&pair);
        FP12_BN254_mul(&t, &pair);//配对结果乘以群上单位元
        cfe_string_free(&label_for_hash);
        cfe_string_free(&str_i);
    }
    FP12_BN254_inv(&t, &t);
    FP12_BN254_mul(&s, &t);

    PAIR_BN254_ate(&pair, &gen2, &gen1);
    PAIR_BN254_fexp(&pair);
    //计算边界值，求解离散对数
    mpz_t res_bound;
    mpz_init(res_bound);
    mpz_pow_ui(res_bound, bound, 2);
    mpz_mul_ui(res_bound, res_bound, y->size);

    cfe_error err;
    err = cfe_baby_giant_FP12_BN256_with_neg(res, &s, &pair, res_bound);

    mpz_clears(res_bound, y_i_mod, order, NULL);
    cfe_vec_G2_free(&keys_sum);

    return err;
}