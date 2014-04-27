#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

struct thread_data{
    int base,           //base term from which computation will start
        num_terms;      //numer of terms to compute
    double result;
};

void *worker(void *arg){
    struct thread_data *data = arg;
    
    data->result = 0.0;
    for(int i=0;i<data->num_terms;i++){
        data->result += pow(-1.0,(double)data->base)/(double)(2*data->base+1);
        data->base++;
    }
    
    pthread_exit(NULL);
}

double pi_series(long num_terms, long num_threads){    
    //Parallelization is performed using manager/worker model
    //TODO:would be nice to check the limit of # of threads
    //ulimit -a | grep "max user processes"
    
    int q = num_terms / num_threads;        //quotient
    int r = num_terms % num_threads;        //remainder
    int base = 0;                           //term base (n)
    double sum = 0.0;                       //sum that we need to compute
    
    pthread_t *thread;
    struct thread_data *data;
    
    //allocate space for threads and their data
    thread = malloc(num_threads*sizeof(*thread));
    data = malloc(num_threads*sizeof(*data));
    
    for(unsigned long i=0;i<num_threads;i++){
        data[i].base = base;
        data[i].num_terms = q;
        if(r > 0){
            data[i].num_terms++; r--;
        }

        //create only as many threads as we need for current workload
        if (data[i].num_terms > 0)
            pthread_create(thread+i, NULL, worker, data+i);
        
        //should be thread safe, since it's a read operation and value shouldn't change
        base += data[i].num_terms;
    }
    
    for(unsigned long i=0;i<num_threads;i++){
        //this subroutine blocks the calling thread until the specified thread terminates
        //which means that main wont finish until all workers are done with their routines
        pthread_join(thread[i], NULL);
        sum += data[i].result;
    }        

    //uuu... there is no garbage collector :P
//    free(thread);
//    free(data);

    return 4*sum;
}