#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

struct thread_data{
    int base,           //base term from which computation will start
        num_elements;   //numer of terms to compute
    double result;
};

void *worker(void *arg){
    struct thread_data *data = arg;
    
    for(int i=0;i<data->num_elements;i++){
        data->result += 1.0/((double) data->base*data->base);
        data->base++;
    }
    
    pthread_exit(NULL);
}

double pi_series(long num_terms, long num_threads){    
    //Parallelization is performed using manager/worker model
    //TODO:would be nice to check the limit of # of threads
    //ulimit -a | grep "max user processes"
    
    int q = (int)num_terms / num_threads;   //quotient
    int r = num_terms % num_threads;        //remainder
    int base = 1;                           //term base (n)
    double sum = 0.0;                       //sum that we need to compute
    
    pthread_t *thread;
    struct thread_data *data;
    thread = malloc(num_threads*sizeof(*thread));
    data = malloc(num_threads*sizeof(*data));
    
    
    for(unsigned long i=0;i<num_threads;i++){
        data[i].result = 0.0;
        data[i].base = base;
        data[i].num_elements = q;
        (r > 0) ? data[i].num_elements++ : r--;
        
        pthread_create(thread+i, NULL, worker, data+i);
    }
    
    for(unsigned long i=0;i<num_threads;i++){
        //this subroutine blocks the calling thread until the specified thread terminates
        pthread_join(thread[i], NULL);
        sum += data[i].result;
    }        

    //uuu... there is no garbage collector :P
    for (unsigned long i=0;i<num_threads;i++){
        free(&thread[i]);
        free(&data[i]);
    }

    return sqrt(6*sum);
}