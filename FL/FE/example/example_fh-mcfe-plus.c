#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>       
#include <sodium.h>     
#include <amcl/pair_BN254.h>
#include "../include/cifer/innerprod/fullysec/fh_multi_ipe_plus.h"
#include "cifer/sample/uniform.h"


double get_elapsed_time(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + 
           (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

void print_vec(const char* label, cfe_vec* v) {
    printf("%s: [", label);
    for (size_t i=0; i<v->size; i++) {
        mpz_out_str(stdout, 10, v->vec[i]);
        if (i != v->size-1) printf(", ");
    }
    printf("]\n");
}

void test_fh_multi_ipe() {
    struct timespec start, end;
    if (sodium_init() < 0) {
        fprintf(stderr, "error\n");
        exit(EXIT_FAILURE);
    }
    size_t sec_level = 2;
    size_t num_clients = 10;
    size_t vec_len = 2;
    mpz_t bound_x, bound_y;
    mpz_inits(bound_x, bound_y, NULL);
    mpz_set_ui(bound_x, 10000);
    mpz_set_ui(bound_y, 10000);
    cfe_fh_multi_ipe scheme;
    cfe_error err = cfe_fh_multi_ipe_plus_init(&scheme, sec_level, num_clients, vec_len, bound_x, bound_y);
    clock_gettime(CLOCK_MONOTONIC, &start);
    cfe_fh_multi_ipe_sec_key master_key;
    FP12_BN254 pub_key;
    cfe_fh_multi_ipe_master_key_plus_init(&master_key, &scheme);
    err = cfe_fh_multi_ipe_generate_plus_keys(&master_key, &pub_key, &scheme);
    clock_gettime(CLOCK_MONOTONIC, &end);    
    cfe_vec x[num_clients];
    cfe_vec eta;
    int eta_idx = 0;
    cfe_vec_init(&eta, vec_len); 
    mpz_t mask[num_clients];
    mpz_t lower, upper, mask_sum;
    mpz_inits(lower, upper, mask_sum, NULL);
    mpz_set_si(lower, 0);
    mpz_set(upper, scheme.bound_x);
    for (size_t i = 0; i < num_clients; i++) {
        cfe_vec_init(&x[i], vec_len);                    
        mpz_init(mask[i]);                             
        cfe_uniform_sample_range_vec(&x[i], lower, upper);  
        cfe_uniform_sample_range(mask[i], lower, upper);   
        if(i==eta_idx){
            cfe_vec_copy(&eta, &x[eta_idx]);
        }
        printf("Client %zu: ", i);
        print_vec("x", &x[i]);
    }
    mpz_set_ui(mask_sum, 0); 
    for (size_t i = 0; i < num_clients; i++) {
        mpz_add(mask_sum, mask_sum, mask[i]); 
    }
    cfe_vec_G1 ciphers[num_clients];
    double total_encrypt_time = 0.0;
    
    for (size_t i=0; i<num_clients; i++) {
        cfe_fh_multi_ipe_ciphertext_plus_init(&ciphers[i], &scheme);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        err = cfe_fh_multi_ipe_plus_encrypt(&ciphers[i], &x[i], &master_key.B_hat[i], &scheme, mask[i]);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        total_encrypt_time += get_elapsed_time(&start, &end);
    }
    mpz_t expected_sums[vec_len];
    mpz_t decrypted_sums[vec_len]; 
    for (size_t k=0; k<vec_len; k++) {
        mpz_init(expected_sums[k]);
        mpz_init(decrypted_sums[k]);
        mpz_set_ui(expected_sums[k], 0);
        for (size_t i=0; i<num_clients; i++) {
            mpz_add(expected_sums[k], expected_sums[k], x[i].vec[k]);
        }
    }

    double total_fe_key_time = 0.0;  
    double total_decrypt_time = 0.0;
    
    for (size_t k=0; k<vec_len; k++) {
        cfe_mat y;
        cfe_mat_init(&y, num_clients, vec_len);
        for (size_t i=0; i<num_clients; i++) {
            for (size_t j=0; j<vec_len; j++) {
                mpz_set_ui(y.mat[i].vec[j], (j == k) ? 1 : 0);
            }
        }

        cfe_mat_G2 fe_key;
        cfe_fh_multi_ipe_fe_key_plus_init(&fe_key, &scheme);
        clock_gettime(CLOCK_MONOTONIC, &start);
        err = cfe_fh_multi_ipe_derive_fe_plus_key(&fe_key, &y, &master_key, &scheme);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double fe_time = get_elapsed_time(&start, &end);
        total_fe_key_time += fe_time;

        mpz_t decrypted_sum;
        mpz_init(decrypted_sum);
        clock_gettime(CLOCK_MONOTONIC, &start);
        err = cfe_fh_multi_ipe_plus_decrypt(decrypted_sum, ciphers, &fe_key, &pub_key, &scheme, mask_sum);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double decrypt_time = get_elapsed_time(&start, &end);
        total_decrypt_time += decrypt_time;

        mpz_set(decrypted_sums[k], decrypted_sum);
        mpz_clear(decrypted_sum);
        cfe_mat_G2_free(&fe_key);
        cfe_mat_free(&y);
    }

    double cos_key_gen_time = 0.0;
    double cos_decrypt_time = 0.0;
    cfe_mat_G2 cos_key;
    size_t idx = 1;
    clock_gettime(CLOCK_MONOTONIC, &start);
    err = cfe_fh_multi_ipe_cos_derive_plus_key(&cos_key, idx, &eta, &master_key, &scheme);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cos_key_gen_time = get_elapsed_time(&start, &end);
    mpz_t result;
    mpz_init(result);
    clock_gettime(CLOCK_MONOTONIC, &start);
    err = cfe_fh_multi_ipe_cos_plus_decrypt(result, &ciphers[idx], &cos_key, 
                                    &pub_key, &scheme);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cos_decrypt_time = get_elapsed_time(&start, &end);
    if (err == CFE_ERR_NONE) {
        gmp_printf("inner product of client %zu and client %d: %Zd\n", idx, eta_idx, result);
    } else {
    }

    // 打印向量化解密结果
    printf("\nresult of aggregation: [");
    for (size_t k=0; k<vec_len; k++) {
        mpz_out_str(stdout, 10, decrypted_sums[k]);
        if (k != vec_len-1) printf(", ");
    }
    printf("]\n");

    int all_correct = 1;
    for (size_t k=0; k<vec_len; k++) {
        if (mpz_cmp(decrypted_sums[k], expected_sums[k]) != 0) {
            all_correct = 0;
            gmp_printf("fault", k, expected_sums[k], decrypted_sums[k]);
        }
    }
    for (size_t k=0; k<vec_len; k++) {
        mpz_clear(expected_sums[k]);
        mpz_clear(decrypted_sums[k]);
    }
    for (size_t i=0; i<num_clients; i++) {
        cfe_vec_free(&x[i]);
         mpz_clear(mask[i]);
        cfe_vec_G1_free(&ciphers[i]);
    }
    cfe_fh_multi_ipe_master_key_plus_free(&master_key);
    cfe_fh_multi_ipe_plus_free(&scheme);
    mpz_clears(bound_x, bound_y, lower, upper, mask_sum, NULL);
}

int main() {
    test_fh_multi_ipe();
    return 0;
}