#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define H 32
#define W 32
#define IN_C 1
#define OUT_C 8
#define K 3
#define PAD 1
#define STRIDE 1

#define OUT_H ((H + 2*PAD - K)/STRIDE + 1)
#define OUT_W ((W + 2*PAD - K)/STRIDE + 1)

#define PRINT_N 10

// -----------------------
// CPU version (naive conv)
// -----------------------
float cpu_conv(
    float* input,       // [IN_C, H, W]
    float* filter,      // [OUT_C, IN_C, K, K]
    float* output       // [OUT_C, OUT_H, OUT_W]
){
    clock_t start = clock();

    for(int oc=0; oc<OUT_C; oc++){
        for(int oh=0; oh<OUT_H; oh++){
            for(int ow=0; ow<OUT_W; ow++){
                float sum = 0;
                for(int ic=0; ic<IN_C; ic++){
                    for(int kh=0; kh<K; kh++){
                        for(int kw=0; kw<K; kw++){

                            int ih = oh*STRIDE + kh - PAD;
                            int iw = ow*STRIDE + kw - PAD;

                            float v = 0;
                            if(ih>=0 && ih<H && iw>=0 && iw<W)
                                v = input[ic*H*W + ih*W + iw];

                            float w = filter[oc*IN_C*K*K + ic*K*K + kh*K + kw];

                            sum += v * w;
                        }
                    }
                }
                output[oc*OUT_H*OUT_W + oh*OUT_W + ow] = sum;
            }
        }
    }

    clock_t end = clock();
    return ((double)(end - start)/CLOCKS_PER_SEC)*1000.0;
}

// -----------------------
// im2col
// -----------------------
void im2col(float* input, float* X_col)
{
    int col_idx = 0;
    for(int oh=0; oh<OUT_H; oh++){
        for(int ow=0; ow<OUT_W; ow++){
            for(int ic=0; ic<IN_C; ic++){
                for(int kh=0; kh<K; kh++){
                    for(int kw=0; kw<K; kw++){

                        int ih = oh*STRIDE + kh - PAD;
                        int iw = ow*STRIDE + kw - PAD;

                        float v = 0;
                        if(ih>=0 && ih<H && iw>=0 && iw<W)
                            v = input[ic*H*W + ih*W + iw];

                        X_col[(ic*K*K + kh*K + kw) * (OUT_H*OUT_W) + col_idx] = v;
                    }
                }
            }
            col_idx++;
        }
    }
}

// -----------------------
// Main
// -----------------------
int main()
{
    printf("2D Convolution using cuBLAS SGEMM\n");
    printf("Input  : %d x %d x %d\n", IN_C, H, W);
    printf("Filter : %d x %d x %d x %d\n", OUT_C, IN_C, K, K);
    printf("Output : %d x %d x %d\n", OUT_C, OUT_H, OUT_W);

    int input_size  = IN_C * H * W;
    int filter_size = OUT_C * IN_C * K * K;
    int out_size    = OUT_C * OUT_H * OUT_W;
    int xcol_size   = (IN_C*K*K) * (OUT_H*OUT_W);
    int wcol_size   = OUT_C * (IN_C*K*K);

    float *h_input  = (float*)malloc(sizeof(float)*input_size);
    float *h_filter = (float*)malloc(sizeof(float)*filter_size);
    float *h_output_cpu = (float*)malloc(sizeof(float)*out_size);
    float *h_output_gpu = (float*)malloc(sizeof(float)*out_size);
    float *X_col = (float*)malloc(sizeof(float)*xcol_size);

    for(int i=0;i<input_size;i++) h_input[i] = rand()%5;
    for(int i=0;i<filter_size;i++) h_filter[i] = (rand()%5)-2;
    for(int i=0;i<out_size;i++) h_output_cpu[i] = 0;

    // CPU version
    float cpu_time = cpu_conv(h_input, h_filter, h_output_cpu);

    // make W_col
    float *W_col = (float*)malloc(sizeof(float)*wcol_size);
    for(int oc=0; oc<OUT_C; oc++){
        for(int ic=0; ic<IN_C; ic++){
            for(int kh=0; kh<K; kh++){
                for(int kw=0; kw<K; kw++){
                    int r = oc;  
                    int c = ic*K*K + kh*K + kw;
                    W_col[r*(IN_C*K*K) + c] =
                        h_filter[oc*IN_C*K*K + ic*K*K + kh*K + kw];
                }
            }
        }
    }

    im2col(h_input, X_col);

    // Allocate device memory
    float *d_W, *d_X, *d_Y;
    cudaMalloc((void**)&d_W, sizeof(float)*wcol_size);
    cudaMalloc((void**)&d_X, sizeof(float)*xcol_size);
    cudaMalloc((void**)&d_Y, sizeof(float)*out_size);

    // Copy
    cudaMemcpy(d_W, W_col, sizeof(float)*wcol_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X_col, sizeof(float)*xcol_size, cudaMemcpyHostToDevice);

    // cuBLAS SGEMM
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        OUT_H*OUT_W,        // m
        OUT_C,              // n
        IN_C*K*K,           // k
        &alpha,
        d_X, OUT_H*OUT_W,
        d_W, IN_C*K*K,
        &beta,
        d_Y, OUT_H*OUT_W
    );
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(h_output_gpu, d_Y, sizeof(float)*out_size, cudaMemcpyDeviceToHost);

    // ---------------------
    // Error Norm
    // ---------------------
    double err = 0;
    for(int i=0;i<out_size;i++){
        double d = h_output_cpu[i] - h_output_gpu[i];
        err += d*d;
    }

    printf("\n[Performance Result]\n");
    printf("- CPU Time: %f ms\n", cpu_time);
    printf("- GPU Time: %f ms\n", gpu_time);
    printf("- Speedup : %fx\n", cpu_time / gpu_time);
    printf("- Total Error Norm: %e\n", sqrt(err));
    printf("--------------------------------\n\n");

    printf("[Sample Output Comparison]\n");
    for(int i=0;i<PRINT_N;i++){
        printf("CPU[%d] = %f, GPU[%d] = %f\n",
            i, h_output_cpu[i], i, h_output_gpu[i]);
    }

    return 0;
}
