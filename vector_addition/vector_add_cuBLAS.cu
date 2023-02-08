//!nvcc -o vector vector.cu -lcublas
#include <iostream>
#include <random>
#include <cmath>
#include <cublas_v2.h>

enum {FAIL, SUCCESS};

void init_vector(float* vec,const int len)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 99);
    
    for(int i=0;i<len;i++) vec[i] = dis(gen);
}

int verifier(const float* vec1, const float* vec2, const float* res, const float scale ,const int len)
{
    for(int i=0;i<len;i++)
    {
        if(scale * vec1[i]+vec2[i] != res[i]) return FAIL;
    }
    return SUCCESS;
}

int main(void)
{
    int N = 1 << 16;  //vector size (65536)
    float *h_a, *h_b, *h_c; //host vector
    float *d_a, *d_b; //device vector
 
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    cudaMalloc(&d_a,N*sizeof(float));
    cudaMalloc(&d_b,N*sizeof(float));

    //Initialize vectors
    init_vector(h_a,N);
    init_vector(h_b,N);

    //Create cublas handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    //Copy vectors from Host to device
    cublasSetVector(N, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(N, sizeof(float), h_b, 1, d_b, 1);

    //Launch simple saxpy kernel (a*x + y)
    const float scale = 2.0f; //a
    cublasSaxpy(handle, N, &scale, d_a, 1, d_b, 1);

    //Copy the result back to Host
    cublasGetVector(N, sizeof(float), d_b, 1, h_c, 1);

    if(verifier(h_a,h_b,h_c,scale,N) == SUCCESS) std::cout<<"YAY!";
    else std::cout << "Hmm..";

    /*
    for(int i=0;i<N;i++){
      printf("%f %f %f\n",h_a[i],h_b[i],h_c[i]);
    }
    */

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
}