//mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sketch_node_embs_fast.c

#include "mex.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"
#include "limits.h"

double *rand_beta;
mwSize rand_n, rand_m;

double *network;
mwSize num_n, num_m;
mwIndex  *ir, *jc;

double *embs_old;
double *embs_new;

long long K_hash;
long long order_max;
double alpha_Katz;
double weight;


void cws_fast_o1(){
//    efficient implementation with sparse input for efficiency, 0.66s for blogcatalog, while 12s for building a full vector and sketch
    double v_temp, v_min,ind_m;
    long long ind_n, ind_k, ind_emb;
    long long counter=0,counter_bak;
    
    for(int i=0; i<num_n; i++){
        ind_n = i*num_n;
        ind_emb = i*K_hash;
//         printf("sketching node %d\n",i);
        
        counter_bak = counter;
        for(int k=0;k<K_hash;k++){
            v_min=INFINITY;
            ind_m=0;
            ind_k = num_m*k;
            
            for(int j=jc[i]; j<jc[i+1]; j++){
                v_temp = rand_beta[ind_k+ir[counter]]/network[counter];
                if(v_temp<v_min){
                    v_min = v_temp;
                    embs_new[ind_emb+k] = ir[counter];
                }
                counter++;
            }
            
            if (k<K_hash-1) {
//                 printf("node seen: %d\n",counter-counter_bak);
                counter = counter_bak;
                
            }
//             printf("v_min, ind_m: %f, %f\n",v_min,ind_m);
        }
    }
//     printf("counter_o1: %d\n", counter);
}


void cws_fast_recursive(){
    double *vec = (double *)mxCalloc(num_m, sizeof(double)); //a node embedding
    double v_temp, v_min,ind_m;
    long long ind_n, ind_k, ind_emb, ind_s;
    long long counter=0;
    double counter_h=0;
    
    for(int i=0; i<num_n; i++){
        ind_n = i*num_n;
        ind_emb = i*K_hash;
        
// build vec
        memset(vec, 0, num_m*sizeof(double)); //fast reset vec to zeros
        for(int j=jc[i]; j<jc[i+1]; j++) {
            vec[ir[counter]] += network[counter];

            ind_s = ir[counter]*K_hash;
            for(int s=0; s<K_hash; s++){
                
                vec[(long long) (embs_old[ind_s+s])] += weight;

                
            }
            counter++;
            
        }

        
        for(int k=0;k<K_hash;k++){
            v_min=INFINITY;
            ind_k = num_m*k;
            
            for(int j=0;j<num_m;j++){
                if(vec[j]!=0){
                    counter_h++;
                    v_temp = rand_beta[ind_k+j]/vec[j];
                    if(v_temp<v_min){
                        v_min = v_temp;
                        embs_new[ind_emb+k] = j;
                    }
                }
            }
        }
    }
//     printf("counter_ox: %f\n",counter_h/K_hash);
}


void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "5 inputs required.");
        //(network, K_hash, Rand_beta, order_max, alpha)
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                "1 output required.");
    }
    
    network = (double *)mxGetData(prhs[0]); // read from file
    num_m = mxGetM(prhs[0]);
    num_n = mxGetN(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    
    
    
    K_hash = mxGetScalar(prhs[1]);
    
    rand_beta = (double *)mxGetData(prhs[2]);
    rand_m = mxGetM(prhs[2]);
    rand_n = mxGetN(prhs[2]);
    
    
    order_max = mxGetScalar(prhs[3]);
    alpha_Katz = mxGetScalar(prhs[4]);
    
    weight = alpha_Katz/K_hash;
    
//     print arguments
    mexPrintf("num of nodes (rows): %lld; num of nodes (columns): %lld; embedding dimension: %lld\n", num_m,num_n, K_hash);
    mexPrintf("rand_m (rows) : %lld; rand_n: %lld\n",  rand_m, rand_n);
    mexPrintf("order_max: %lld\n",order_max);
    mexPrintf("alpha_Katz: %f\n",alpha_Katz);
    fflush(stdout);
    
    plhs[0] = mxCreateDoubleMatrix(K_hash,num_n,mxREAL);
    embs_new = mxGetPr(plhs[0]);
    printf("sketching starts ...\n");
    
    cws_fast_o1();
    
    if (order_max>=2){
        for (int i=1;i<order_max;i++){
            embs_old = (double *)mxMalloc(K_hash*num_n*sizeof(double));
            memcpy(embs_old, embs_new, K_hash*num_n*sizeof(double));
            printf("sketching starts (order %d...)\n",i+1);
            cws_fast_recursive();
        }
        
    }
    
    for(int i=0; i<num_n; i++){
        int ind_emb = i*K_hash;
        for(int k=0; k<K_hash; k++){
            embs_new[ind_emb+k]++;
        }
    }
}






