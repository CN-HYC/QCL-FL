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

#include <amcl/pair_BN254.h>
#include <cifer/internal/common.h>
#define CFE_ERR_HASH_FAILED -100
#include <sodium.h> 
#include "cifer/innerprod/fullysec/fh_multi_ipe_plus.h"
#include "cifer/internal/big.h"
#include "cifer/sample/uniform.h"
#include "cifer/internal/dlog.h"

cfe_error cfe_fh_multi_ipe_plus_init(cfe_fh_multi_ipe *c, size_t sec_level,
                                size_t num_clients, size_t vec_len,
                                mpz_t bound_x, mpz_t bound_y) {
    cfe_error err = CFE_ERR_NONE;
    mpz_t check, order;
    mpz_inits(check, order, NULL);
    mpz_from_BIG_256_56(order, (int64_t *) CURVE_Order_BN254);

    mpz_mul(check, bound_x, bound_y);
    mpz_mul_ui(check, check, vec_len * num_clients);

    if (mpz_cmp(check, order) >= 0) {
        err = CFE_ERR_PRECONDITION_FAILED;
        goto cleanup;
    }

    mpz_inits(c->bound_x, c->bound_y, c->order, NULL); 

    c->sec_level = sec_level;
    c->vec_len = vec_len; 
    c->num_clients = num_clients;
    mpz_set(c->bound_x, bound_x);
    mpz_set(c->bound_y, bound_y);
    mpz_set(c->order, order);

    cleanup:
    mpz_clears(check, order, NULL);
    return err;
}

void cfe_fh_multi_ipe_plus_copy(cfe_fh_multi_ipe *res, cfe_fh_multi_ipe *c) {
    mpz_inits(res->bound_x, res->bound_y, res->order, NULL);
    res->sec_level = c->sec_level;
    res->vec_len = c->vec_len;
    res->num_clients = c->num_clients;
    mpz_set(res->bound_x, c->bound_x);
    mpz_set(res->bound_y, c->bound_y);
    mpz_set(res->order, c->order);
}

void cfe_fh_multi_ipe_plus_free(cfe_fh_multi_ipe *c) {
    mpz_clears(c->bound_x, c->bound_y, c->order, NULL);
}

void cfe_fh_multi_ipe_master_key_plus_init(cfe_fh_multi_ipe_sec_key *sec_key,
                                      cfe_fh_multi_ipe *c) {
    sec_key->B_hat = (cfe_mat *) cfe_malloc(sizeof(cfe_mat) * c->num_clients);
    sec_key->B_star_hat = (cfe_mat *) cfe_malloc(sizeof(cfe_mat) * c->num_clients);
    for (size_t i = 0; i < c->num_clients; i++) {
        cfe_mat_init(&sec_key->B_hat[i], c->vec_len + c->sec_level + 1, 2 * c->vec_len + 2 * c->sec_level + 1);
        cfe_mat_init(&(sec_key->B_star_hat)[i], c->vec_len + c->sec_level, 2 * c->vec_len + 2 * c->sec_level + 1);
    }
    sec_key->num_clients = c->num_clients;
} 

void cfe_fh_multi_ipe_master_key_plus_free(cfe_fh_multi_ipe_sec_key *sec_key) {
    for (size_t i = 0; i < sec_key->num_clients; i++) {
        cfe_mat_frees(&(sec_key->B_hat)[i], &(sec_key->B_star_hat)[i], NULL);
    }
    free(sec_key->B_hat);
    free(sec_key->B_star_hat);
}

// cfe_fh_multi_ipe_random_OB is a helping function used
// in cfe_fh_multi_ipe_generate_keys.
cfe_error cfe_fh_multi_ipe_random_plus_OB(cfe_mat *B, cfe_mat *B_star, mpz_t mu, mpz_t p) {
    cfe_uniform_sample_mat(B, p);
    cfe_mat B_inv;
    cfe_mat_init(&B_inv, B->rows, B->cols);

    cfe_error err = cfe_mat_inverse_mod_gauss(&B_inv, NULL, B, p);
    if (err != CFE_ERR_NONE) {
        goto cleanup;
    }

    cfe_mat_transpose(B_star, &B_inv);
    cfe_mat_mul_scalar(B_star, B_star, mu);
    cfe_mat_mod(B_star, B_star, p);

    cleanup:
    cfe_mat_free(&B_inv);

    return err;
}

cfe_error cfe_fh_multi_ipe_generate_plus_keys(cfe_fh_multi_ipe_sec_key *sec_key,
                                         FP12_BN254 *pub_key, cfe_fh_multi_ipe *c) {
    cfe_error err = CFE_ERR_NONE;
    mpz_t mu;
    BIG_256_56 mu_big;
    mpz_init(mu);
    cfe_uniform_sample(mu, c->order);
    BIG_256_56_from_mpz(mu_big, mu);

    ECP_BN254 g1;
    ECP2_BN254 g2;
    ECP_BN254_generator(&g1);
    ECP2_BN254_generator(&g2);
    PAIR_BN254_ate(pub_key, &g2, &g1);
    PAIR_BN254_fexp(pub_key);
    FP12_BN254_pow(pub_key, pub_key, mu_big);

    cfe_mat *B = (cfe_mat *) cfe_malloc(sizeof(cfe_mat) * c->num_clients);
    cfe_mat *B_star = (cfe_mat *) cfe_malloc(sizeof(cfe_mat) * c->num_clients);
    for (size_t i = 0; i < c->num_clients; i++) {
        cfe_mat_init(&B[i], 2 * c->vec_len + 2 * c->sec_level + 1, 2 * c->vec_len + 2 * c->sec_level + 1);
        cfe_mat_init(&B_star[i], 2 * c->vec_len + 2 * c->sec_level + 1, 2 * c->vec_len + 2 * c->sec_level + 1);
    }

    for (size_t i = 0; i < c->num_clients; i++) {
        err = cfe_fh_multi_ipe_random_OB(&B[i], &B_star[i], mu, c->order);
        if (err != CFE_ERR_NONE) {
            goto cleanup;
        }

        for (size_t j = 0; j < c->vec_len + c->sec_level + 1; j++) {
            if (j < c->vec_len) {
                cfe_vec_copy(&sec_key->B_hat[i].mat[j], &B[i].mat[j]);
                cfe_vec_copy(&sec_key->B_star_hat[i].mat[j], &B_star[i].mat[j]);
            } else if (j == c->vec_len) {
                cfe_vec_copy(&sec_key->B_hat[i].mat[j], &B[i].mat[j + c->vec_len]);
                cfe_vec_copy(&sec_key->B_star_hat[i].mat[j], &B_star[i].mat[j + c->vec_len]);
            } else if (j < c->vec_len + c->sec_level) {
                cfe_vec_copy(&sec_key->B_hat[i].mat[j], &B[i].mat[j - 1 + c->vec_len + c->sec_level]);
                cfe_vec_copy(&sec_key->B_star_hat[i].mat[j], &B_star[i].mat[j + c->vec_len]);
            } else {
                cfe_vec_copy(&sec_key->B_hat[i].mat[j], &B[i].mat[j - 1 + c->vec_len + c->sec_level]);
            }
        }
    }

    cleanup:
    mpz_clear(mu);
    for (size_t i = 0; i < c->num_clients; i++) {
        cfe_mat_frees(&B[i], &B_star[i], NULL);
    }
    free(B);
    free(B_star);

    return err;
}

void cfe_fh_multi_ipe_fe_key_plus_init(cfe_mat_G2 *fe_key, cfe_fh_multi_ipe *c) {
    cfe_mat_G2_init(fe_key, c->num_clients, 2 * c->vec_len + 2 * c->sec_level + 1);
}

cfe_error cfe_fh_multi_ipe_derive_fe_plus_key(cfe_mat_G2 *fe_key, cfe_mat *y,
                                        cfe_fh_multi_ipe_sec_key *sec_key,
                                        cfe_fh_multi_ipe *c) {
    if (!cfe_mat_check_bound(y, c->bound_y)) {
        return CFE_ERR_BOUND_CHECK_FAILED;
    }

    mpz_t zero, s;
    mpz_inits(zero, s, NULL);
    mpz_set_ui(zero, 0);
    cfe_mat gamma, key_mat;
    cfe_mat_init(&gamma, c->sec_level-1, c->num_clients);
    cfe_uniform_sample_mat(&gamma, c->order);

    cfe_mat_init(&key_mat, c->num_clients, 2 * c->vec_len + 2 * c->sec_level + 1);
    cfe_mat_set_const(&key_mat, zero); 
    cfe_vec tmp_vec; 
    cfe_vec_init(&tmp_vec, 2 * c->vec_len + 2 * c->sec_level + 1);

    for (size_t i = 0; i < c->num_clients; i++) {
        for (size_t j = 0; j < c->vec_len + c->sec_level - 1; j++) {
            if (j < c->vec_len) {
                cfe_mat_get(s, y, i, j);
            }
            else if (j == c->vec_len) {
                mpz_set_ui(s, 1);
            } else {
                cfe_mat_get(s, &gamma, j - c->vec_len, i);
            }

            cfe_vec_mul_scalar(&tmp_vec, &sec_key->B_star_hat[i].mat[j], s);
            cfe_vec_add(&key_mat.mat[i], &key_mat.mat[i], &tmp_vec);
            cfe_vec_mod(&key_mat.mat[i], &key_mat.mat[i], c->order);
        }
    }

    cfe_mat_mul_G2(fe_key, &key_mat);

    cfe_vec_free(&tmp_vec);
    cfe_mat_frees(&key_mat, &gamma, NULL);
    mpz_clears(zero, s, NULL);

    return CFE_ERR_NONE;
}


cfe_error cfe_fh_multi_ipe_cos_derive_plus_key(cfe_mat_G2 *cos_key, size_t idx,
                                          cfe_vec *eta,
                                          cfe_fh_multi_ipe_sec_key *sec_key,
                                          cfe_fh_multi_ipe *c) {

    cfe_mat_G2_init(cos_key, 1, 2 * c->vec_len + 2 * c->sec_level + 1);
    mpz_t s;
    mpz_init(s);
    cfe_mat key_mat;
    cfe_mat_init(&key_mat, 1, 2 * c->vec_len + 2 * c->sec_level + 1);
    cfe_vec tmp_vec;
    cfe_vec_init(&tmp_vec, 2 * c->vec_len + 2 * c->sec_level + 1);

    mpz_t zero;
    mpz_init_set_ui(zero, 0);
    cfe_mat_set_const(&key_mat, zero);
    mpz_clear(zero);

    cfe_mat gamma;
    cfe_mat_init(&gamma, c->sec_level - 1, c->num_clients);
    cfe_uniform_sample_mat(&gamma, c->order); 

   for (size_t j = 0; j < c->vec_len + c->sec_level - 1; j++) {
    if (j < c->vec_len) {
        mpz_set(s, eta->vec[j]); 
    } else if (j == c->vec_len) {
        mpz_set_ui(s, 0);
    } else {
        mpz_set(s, gamma.mat[j - c->vec_len].vec[idx]);
    }

    cfe_vec_mul_scalar(&tmp_vec, &sec_key->B_star_hat[idx].mat[j], s);
    cfe_vec_add(&key_mat.mat[0], &key_mat.mat[0], &tmp_vec);
    cfe_vec_mod(&key_mat.mat[0], &key_mat.mat[0], c->order);
    }

    cfe_mat_free(&gamma);

    cfe_mat_mul_G2(cos_key, &key_mat);

    cfe_vec_free(&tmp_vec);
    cfe_mat_free(&key_mat);
    mpz_clear(s);

    return CFE_ERR_NONE;
}



void cfe_fh_multi_ipe_ciphertext_plus_init(cfe_vec_G1 *cipher, cfe_fh_multi_ipe *c) {
    cfe_vec_G1_init(cipher, 2 * c->vec_len + 2 * c->sec_level + 1);
}

cfe_error cfe_fh_multi_ipe_plus_encrypt(cfe_vec_G1 *cipher, cfe_vec *x,
                                   cfe_mat *part_sec_key, cfe_fh_multi_ipe *c, mpz_t r) {
    if (!cfe_vec_check_bound(x, c->bound_x)) {
        return CFE_ERR_BOUND_CHECK_FAILED;
    }

    mpz_t zero, s;
    mpz_inits(zero, s, NULL);
    mpz_set_ui(zero, 0);
    cfe_vec phi, key_vec;
    cfe_vec_init(&phi, c->sec_level);
    cfe_uniform_sample_vec(&phi, c->order);

    cfe_vec_init(&key_vec, 2 * c->vec_len + 2 * c->sec_level + 1);
    cfe_vec_set_const(&key_vec, zero);
    cfe_vec tmp_vec;
    cfe_vec_init(&tmp_vec, 2 * c->vec_len + 2 * c->sec_level + 1);

    for (size_t j = 0; j < c->vec_len + c->sec_level + 1; j++) {
        if (j < c->vec_len) {
            cfe_vec_get(s, x, j);
        } else if (j == c->vec_len) {
            mpz_set(s, r);
        } else {
            cfe_vec_get(s, &phi, j - c->vec_len - 1);
        }

        cfe_vec_mul_scalar(&tmp_vec, &part_sec_key->mat[j], s);
        cfe_vec_add(&key_vec, &key_vec, &tmp_vec);
        cfe_vec_mod(&key_vec, &key_vec, c->order);
    }

    cfe_vec_mul_G1(cipher, &key_vec);

    cfe_vec_frees(&tmp_vec, &key_vec, &phi, NULL);
    mpz_clears(zero, s, NULL);

    return CFE_ERR_NONE;
}

cfe_error cfe_fh_multi_ipe_plus_decrypt(mpz_t res, cfe_vec_G1 *ciphers, cfe_mat_G2 *fe_key,
                                   FP12_BN254 *pub_key, cfe_fh_multi_ipe *c, mpz_t mask) {
    FP12_BN254 sum, paired;
    FP12_BN254_one(&sum);

    for (size_t i = 0; i < c->num_clients; i++) {
        for (size_t j = 0; j < 2 * c->vec_len + 2 * c->sec_level + 1; j++) {
            PAIR_BN254_ate(&paired, &fe_key->mat[i].vec[j], &ciphers[i].vec[j]);
            PAIR_BN254_fexp(&paired);
            FP12_BN254_mul(&sum, &paired);
        }
    }

    mpz_t res_bound, temp;
    mpz_init(res_bound);
    mpz_init(temp);
    mpz_mul(res_bound, c->bound_x, c->bound_y);
    mpz_mul_ui(res_bound, res_bound, c->num_clients * c->vec_len);

    cfe_error err = cfe_baby_giant_FP12_BN256_with_neg(res, &sum, pub_key, res_bound);
    mpz_sub(temp, res, mask);
    mpz_set(res, temp);
    mpz_clear(temp);
    mpz_clear(res_bound);

    return err;
}


cfe_error cfe_fh_multi_ipe_cos_plus_decrypt(mpz_t res, cfe_vec_G1 *ct_idx,
                                       cfe_mat_G2 *cos_key,
                                       FP12_BN254 *pub_key, cfe_fh_multi_ipe *c) {

    FP12_BN254 sum, paired;
    FP12_BN254_one(&sum); 

    for (size_t j = 0; j < cos_key->cols; j++) { 
        PAIR_BN254_ate(&paired, &cos_key->mat[0].vec[j], &ct_idx->vec[j]);
        PAIR_BN254_fexp(&paired); 

        FP12_BN254_mul(&sum, &paired);
    }

    mpz_t res_bound;
    mpz_init(res_bound);
    mpz_mul(res_bound, c->bound_x, c->bound_y);
    mpz_mul_ui(res_bound, res_bound, c->vec_len);

    cfe_error err = cfe_baby_giant_FP12_BN256_with_neg(res, &sum, pub_key, res_bound);

    if (err != CFE_ERR_NONE) {
        mpz_clear(res_bound);
        return err;
    }

    mpz_clear(res_bound);

    return err;
}