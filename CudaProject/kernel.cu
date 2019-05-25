#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include "Window.h"

using namespace std;

std::ofstream of;
std::ofstream pos;
std::ofstream lines;

#define CUDA(call, string) cudaStatus = (call); if (cudaStatus != cudaSuccess) {cout << string << " " << cudaGetErrorString(cudaStatus) << endl; return cudaStatus;}
#define CUDAE(call) CUDA(call, "Error")

cudaError_t callCuda(int* data, int count);

int profileCopies(float        *h_a,
	float        *h_b,
	float        *d,
	unsigned int  n,
	char         *desc)
{
	cudaError_t cudaStatus;
	cout << "\n" << desc <<  " transfers\n";

	unsigned int bytes = n * sizeof(float);

	// events for timing
	cudaEvent_t startEvent, stopEvent;

	CUDAE(cudaEventCreate(&startEvent));
	CUDAE(cudaEventCreate(&stopEvent));

	CUDAE(cudaEventRecord(startEvent, 0));
	CUDAE(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
	CUDAE(cudaEventRecord(stopEvent, 0));
	CUDAE(cudaEventSynchronize(stopEvent));

	float time;
	CUDAE(cudaEventElapsedTime(&time, startEvent, stopEvent));
	cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-6 / time << " time: " << time << "\n";

	CUDAE(cudaEventRecord(startEvent, 0));
	CUDAE(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
	CUDAE(cudaEventRecord(stopEvent, 0));
	CUDAE(cudaEventSynchronize(stopEvent));

	CUDAE(cudaEventElapsedTime(&time, startEvent, stopEvent));
	cout << "  Device to Host bandwidth (GB/s): " << bytes * 1e-6 / time << "\n";

	for (int i = 0; i < n; ++i) {
		if (h_a[i] != h_b[i]) {
			cout << "*** " << desc << "transfers failed ***\n";
			break;
		}
	}

	// clean up events
	CUDAE(cudaEventDestroy(startEvent));
	CUDAE(cudaEventDestroy(stopEvent));

	return 0;
}

__device__ void partition(int* lp, int* rp, int pivot, int left, int right) {

}

__global__ void quicksort(int* data, int left, int right) {
	int nLeft, nRight;
	cudaStream_t s1, s2;

	partition(data + left, data + right, data[left], nLeft, nRight);

	if (left < nRight) {
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		quicksort << < 1,1 >> > (data, left, nRight);
	}
	if (nLeft < right) {
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		quicksort << < 1,1 >> > (data, nLeft, right);
	}
}

__global__ void child() {
	printf("Hello\n");
}

__global__ void parent() {
	child << <1, 1 >> > ();
	child << <1, 1 >> > ();
	printf("World\n");
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
	Window window(hInstance, nCmdShow);
	pos.open("data.txt");
	lines.open("lines.txt");
	of.open("console.txt", std::ios_base::app);
	of << "==============New Game==============" << std::endl;

	cudaError_t cudaStatus;
	cout << "start\n";
	{
		const unsigned int N = 1;//1048576;
	const unsigned int bytes = N * sizeof(int);
	int *h_a = new int[N];
	int *d_a;
	cout << "end" << endl;
	cudaMalloc((int**)&d_a, bytes);

	memset(h_a, 0, bytes);
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	CUDAE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	cudaFree(d_a);
	delete[] h_a;
	}

	//example 2
	unsigned int nElements = 1 * 1024 * 1024;
	const unsigned int bytes = nElements * sizeof(float);
	// host arrays
	float *h_aPageable, *h_bPageable;
	float *h_aPinned, *h_bPinned;
	// device array
	float *d_a;
	// allocate and initialize
	h_aPageable = new float[nElements];                    // host pageable
	h_bPageable = new float[nElements];                    // host pageable
	CUDAE(cudaMallocHost((void**)&h_aPinned, bytes)); // host pinned
	CUDAE(cudaMallocHost((void**)&h_bPinned, bytes)); // host pinned
	CUDAE(cudaMalloc((void**)&d_a, bytes));           // device

	for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;

	memcpy(h_aPinned, h_aPageable, bytes);
	memset(h_bPageable, 0, bytes);
	memset(h_bPinned, 0, bytes);

	// output device info and transfer size
	cudaDeviceProp prop;
	CUDAE(cudaGetDeviceProperties(&prop, 0));

	cout << "\nDevice:" << prop.name << "\n";
	cout << "Transfer size (MB): " << bytes / (1024 * 1024) << "\n";

	// perform copies and report bandwidth
	profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
	profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

	// cleanup
	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	delete[] h_aPageable;
	delete[] h_bPageable;

	// call cuda function.
	int* data = nullptr;
	int count = 1024 * 1024;
	data = new int[count];
	cudaStatus = callCuda(data, count);
	if (cudaStatus != cudaSuccess) {
		cout << "call to Cuda failed!";
		return 1;
	}
	delete[] data;


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cout << "cudaDeviceReset failed!";
		return 1;
	}

	parent << <1, 1 >> > ();
	cout << "end" << endl;

	window.MSGLoop();
	of.close();
	pos.close();
	lines.close();

	return 0;
}

cudaError_t callCuda(int* data, int count) {
	cudaError_t cudaStatus;
	int* d_a;
	int bytes = count * 4;
	CUDAE(cudaMalloc((void**)&d_a, bytes));
	CUDAE(cudaMemcpy(d_a, data, bytes, cudaMemcpyHostToDevice));
	quicksort << <1,1 >> > (d_a, 0, count - 1);
	CUDAE(cudaMemcpy(data, d_a, bytes, cudaMemcpyDeviceToHost));
	cudaFree(d_a);

	return cudaSuccess;
}

/*
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/
