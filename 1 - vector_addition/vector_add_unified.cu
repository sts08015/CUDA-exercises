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
    int *a,*b,*c; //unified memory pointers
  
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    init_vector(a,N);
    init_vector(b,N);

    int devId;
    cudaGetDevice(&devId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, devId);

    const int NUM_THREADS = props.maxThreadsPerBlock;
    const int NUM_BLOCKS = props.multiProcessorCount;

    /*
      Unified memory gets fetched on-demand by page fault.
      However, if a programmer explicitly writes cudaMemPrefetchAsync when page fault is obvious, 
      the unified memory gets transferred ahead of time, which can improve performance.
    */
    cudaMemPrefetchAsync(a,size,devId); //Host to Device
    cudaMemPrefetchAsync(b,size,devId); //Host to Device
    cudaMemPrefetchAsync(c,size,devId); //Host to Device
    vectorAdd<<<NUM_BLOCKS,NUM_THREADS>>>(a,b,c,N);
    
    cudaDeviceSynchronize();  //prevents race condition

    cudaMemPrefetchAsync(c,size,cudaCpuDeviceId); //Device to Host
    if(verifier(a,b,c,N) == SUCCESS) std::cout<<"YAY!";
    else std::cout << "Hmm..";

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}