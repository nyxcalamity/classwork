#include <math.h>

double pi_series(long num_terms, long num_threads){
    double sum = 0.0;
    
    
    
    for (unsigned long n = 1; n <= num_terms; n++){
            sum += 1.0 / ((double)n * n);
    }

    return sqrt(6 * sum);
}
