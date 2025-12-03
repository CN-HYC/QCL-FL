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

#define _POSIX_C_SOURCE 199309L // Define POSIX source version for CLOCK_MONOTONIC

#include <stdio.h>
#include <time.h>
#include <sodium.h>
#include "cifer/test.h"
#include "cifer/internal/common.h"
#include "cifer/innerprod/fullysec/dmcfe.h"
#include "cifer/sample/uniform.h"


// Function prototypes
void print_vector(const char *label, cfe_vec *v, size_t l);
double get_time_diff(struct timespec start, struct timespec end);

// Function to print a vector
void print_vector(const char *label, cfe_vec *v, size_t l) {
    printf("%s: [", label);
    for (size_t i = 0; i < l; ++i) {
        mpz_t val;
        mpz_init(val);
        cfe_vec_get(val, v, i);
        gmp_printf("%Zd", val);
        if (i < l - 1) {
            printf(", ");
        }
        mpz_clear(val);
    }
    printf("]\n");
}

// Function to calculate time difference in seconds
double get_time_diff(struct timespec start, struct timespec end) {
    double diff_sec = end.tv_sec - start.tv_sec;
    double diff_nsec = end.tv_nsec - start.tv_nsec;
    return diff_sec + diff_nsec / 1e9;
}

int test_dmcfe_end_to_end() {
    size_t num_clients = 10;
    size_t vector_length = 10; // Length of vectors X and Y
    mpz_t bound, xy_check, xy;
    mpz_inits(bound, xy_check, xy, NULL);
    mpz_set_ui(bound, 2);
    mpz_pow_ui(bound, bound, 13);

    // Initialize the random number generator manually
    if (sodium_init() < 0) {
        fprintf(stderr, "Error initializing sodium\n");
        return 1;
    }

    // Create clients and make an array of their public keys
    cfe_dmcfe_client clients[num_clients];
    ECP_BN254 pub_keys[num_clients];

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < num_clients; i++) {
        cfe_dmcfe_client_init(&(clients[i]), i);
        pub_keys[i] = clients[i].client_pub_key;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Client initialization took %.6f seconds\n", get_time_diff(start, end));

    // Based on public values of each client create private matrices T_i summing to 0
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < num_clients; i++) {
        cfe_dmcfe_set_share(&(clients[i]), pub_keys, num_clients);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Setting shares took %.6f seconds\n", get_time_diff(start, end));

    // Now that the clients have agreed on secret keys they can encrypt a vector in
    // a decentralized way and create partial keys such that only with all of them
    // the decryption of the inner product is possible
    cfe_vec x, y;
    cfe_vec_inits(vector_length, &x, &y, NULL);
    cfe_uniform_sample_vec(&x, bound); // Sample X from [0, 2^10)
    cfe_uniform_sample_vec(&y, bound); // Sample Y from [0, 2^10)

    // Print the randomly generated vectors X and Y
    print_vector("Randomly generated vector X", &x, vector_length);
    print_vector("Randomly generated vector Y", &y, vector_length);

    char label[] = "some label";
    size_t label_len = strlen(label); // length of the label string
    ECP_BN254 ciphers[num_clients];
    cfe_vec_G2 fe_key[num_clients];

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < num_clients; i++) {
        cfe_dmcfe_encrypt(&(ciphers[i]), &(clients[i]), x.vec[i], label, label_len);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Encryption took %.6f seconds\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < num_clients; i++) {
        cfe_dmcfe_fe_key_part_init(&(fe_key[i]));
        cfe_dmcfe_derive_fe_key_part(&(fe_key[i]), &(clients[i]), &y);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Key derivation took %.6f seconds\n", get_time_diff(start, end));

    // Decrypt the inner product with the corresponding label
    clock_gettime(CLOCK_MONOTONIC, &start);
    cfe_error err_dec = cfe_dmcfe_decrypt(xy, ciphers, fe_key, label, label_len, &y, bound);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Decryption took %.6f seconds\n", get_time_diff(start, end));
    if (err_dec != CFE_ERR_NONE) {
        fprintf(stderr, "Error decrypting ciphertext: %d\n", err_dec);
        return 1;
    }

    // Check correctness
    cfe_vec_dot(xy_check, &x, &y);
    if (mpz_cmp(xy, xy_check) != 0) {
        fprintf(stderr, "Decrypted result does not match expected value\n");
        gmp_printf("Expected: %Zd\n", xy_check);
        gmp_printf("Got: %Zd\n", xy);
        return 1;
    }

    // Print the decrypted result
    gmp_printf("Decrypted result: %Zd\n", xy);

    // Free the memory
    mpz_clears(bound, xy_check, xy, NULL);

    for (size_t i = 0; i < num_clients; i++) {
        cfe_dmcfe_client_free(&(clients[i]));
        cfe_vec_G2_free(&(fe_key[i]));
    }
    cfe_vec_frees(&x, &y, NULL);

    return 0;
}

int main(int argc, char *argv[]) {
    if (cfe_init()) {
        perror("Insufficient entropy available for random generation\n");
        return CFE_ERR_INIT;
    }

    int result = test_dmcfe_end_to_end();

    if (result == 0) {
        printf("DMCFE test passed.\n");
    } else {
        printf("DMCFE test failed.\n");
    }

    return result;
}



