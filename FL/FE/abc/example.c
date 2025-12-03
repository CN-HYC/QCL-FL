#include "example.h"
#include <stdio.h>

int add(int a, int b) {
    return a + b + 1;
}

void main(){
    int a = 5, b = 3;
    int result;
    result = add(a,b);
    printf("%d\n", result);
}