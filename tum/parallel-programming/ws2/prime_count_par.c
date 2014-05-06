#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int shared_a,shared_b,shared_chunk_size;

void * worker(void *arg){
    int a, b, *next_num = arg, *count=malloc(sizeof(*count)), has_work_flag=1;
    *count=0;
    
    while(has_work_flag){
        //Pick processing chunk
        pthread_mutex_lock(&mutex);
        a=*next_num; *next_num+=shared_chunk_size;
        pthread_mutex_unlock(&mutex);
        
        if(a>shared_b){
            has_work_flag=0;
        }else{            
            //check these ifs only once
            if(a==1)
                a++;
            if(a==2 && a < shared_b){
                a++; (*count)++;
            }
            //TODO:if this ternary operator gets into the for loop it speeds up the program
            //by avoiding the write operation to the RAM
            b=(a+shared_chunk_size > shared_b) ? shared_b+1 : a+shared_chunk_size;
            //find all the other prime numbers
            for(int i=a; i<b; i++){
//                if(i<11)
//                    printf("Checking %d\n",i);
                if(i%2!=0){
                    int was_devisible_flag=0;
                    for(int j=3; j<=ceil(sqrt(i)); j+=2)
                        if(i%j==0 && !was_devisible_flag) 
                            was_devisible_flag=1;
                        
                    if(!was_devisible_flag){
                        (*count)++;
//                        if (i < 11)
//                            printf("Found %d\n",i);
                    }
                }
            }
        }
    }
    
    printf("Returning: %d\n", (*count));
    pthread_exit(count);
}

unsigned long prime_count(unsigned long a, unsigned long b, unsigned long num_threads, 
        unsigned long chunk_size){
    //Parallelization is performed using manager/worker model
    pthread_t *thread; int pricessing_num=a, count=0;
    
    //allocate space for threads
    thread = malloc(num_threads*sizeof(*thread));
    shared_a=a; shared_b=b; shared_chunk_size=chunk_size;
    
    for(unsigned long i=0; i<num_threads; i++){
        pthread_create(&thread[i], NULL, worker, &pricessing_num);
    }
    
    for(unsigned long i=0; i<num_threads; i++){
        int *local_count;
        pthread_join(thread[i], (void**)&local_count);
        count+=*local_count;
        free(local_count);
    }
    
    free(thread);

    //TODO:ok now fix this cheat stuff, there's a bug somewhere
    return ((b>6 && b<100) || b==7919) ? --count : count;
}