#include <iostream>
#include <random>
#include <cmath>

enum {FAIL, SUCCESS};

__global__ void vectorAdd(int* vec1,int* vec2, int* res, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len) res[idx] = vec1[idx] + vec2[idx];
}

void init_vector(int* vec,const int len)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 99);
    
    for(int i=0;i<len;i++) vec[i] = dis(gen);
}

int verifier(int* vec1, int* vec2, int* res, int len)
{
    for(int i=0;i<len;i++)
    {
        if(vec1[i]+vec2[i] != res[i]) return FAIL;
    }
    return SUCCESS;
}

int main(void)
{
    int N = 1 << 16;  //vector size (65536)
    int size = N*sizeof(int);
    int *h_a, *h_b, *h_c; //host vector
    int *d_a, *d_b, *d_c; //device vector
 
    h_a = new int[N];
    h_b = new int[N];
    h_c = new int[N];
  
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    init_vector(h_a,N);
    init_vector(h_b,N);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    const int NUM_THREADS = 64;
    const int NUM_BLOCKS = (int)ceil((float)N/NUM_THREADS);

    vectorAdd<<<NUM_BLOCKS,NUM_THREADS>>>(d_a,d_b,d_c,N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    if(verifier(h_a,h_b,h_c,N) == SUCCESS) std::cout<<"YAY!";
    else std::cout << "Hmm..";
 
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}