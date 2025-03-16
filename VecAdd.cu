// header files
// standard headers
#include <stdio.h>

// cuda headers
#include<cuda.h>
#include "helper_timer.h"

// global variables
const int iNumberOfArrayElements = 21444777;
//const int iNumberOfArrayElements = 5;

float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold = NULL;

float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

// CUDA kernel
__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{
	// code
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}

// entry-point function
int main(void)
{
	// function declarations
	void fillFloatArrayWithRandomNumbers(float*, int);
	void vecAddCPU(const float*, const float*, float*, int);
	void cleanup(void);

	// variable declarations
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result = cudaSuccess;

	// code
	// host memory allocation
	hostInput1 = (float*)malloc(size);
	if (hostInput1 == NULL)
	{
		printf("Host memory allocation is failed for hostInput1 array.\n");
		cleanup();
		// exit is not portable
		// exit(1) is failure on linux
		// exit(1) is success on openvms
		// Hence for the sake of portability we have used macro
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float*)malloc(size);
	if (hostInput2 == NULL)
	{
		printf("Host memory allocation is failed for hostInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float*)malloc(size);
	if (hostOutput == NULL)
	{
		printf("Host memory allocation is failed for hostOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (float*)malloc(size);
	if (gold == NULL)
	{
		printf("Host memory allocation is failed for gold array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// filling values into host ararys
	fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	// device memory allocation
	// The one * in void** is for host and the other * is for device memory means for VRAM.
	result = cudaMalloc((void**)&deviceInput1, size);
	if (result != cudaSuccess)
	{
		printf("Device memory allcoation is failed for deviceInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceInput2, size);
	if (result != cudaSuccess)
	{
		printf("Device memory allcoatin is failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceOutput, size);
	if (result != cudaSuccess)
	{
		printf("Device memory allcoatin is failed for deviceOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// copy data from host arrays into device arrays
	result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to device data cpy is failed for deviceInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to device data cpy is failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// CUDA kernel configuration
	dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements / 256.0f), 1, 1);
	dim3 dimBlock = dim3(256, 1, 1);

	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// CUDA kernel for vector addtion
	vecAddGPU << <dimGrid, dimBlock >> > (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	// copy data from device arary into host array
	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Devcie to host data copy is failed for hostOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	// vector addition on host
	vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

	// comparison : checking accuracy of data (data in gold(host) and device(GPU)
	const float epsilon = 0.000001f;
	int breakValue = -1;
	bool bAccuracy = true;

	for (int i = 0; i < iNumberOfArrayElements; i++)
	{
		float val1 = gold[i];
		float val2 = hostOutput[i];
		if (fabs(val1 - val2) > epsilon)
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	//char str[128];
	if (bAccuracy == false)
	{
		//sprintf(str, "Comparison of CPU and GPU vector addition is not within accuracy of 0.000001 at array index %d", breakValue);
		printf("Comparison of CPU and GPU vector addition is not within accuracy of 0.000001 at array index %d\n", breakValue);
	}
	else
	{
		//sprintf(str, "Comparison of CPU and GPU vector addition is within accuracy of 0.000001 at array index %d");
		printf("Comparison of CPU and GPU vector addition is within accuracy of 0.000001\n");
	}

	printf("Time on CPU: %.6f\n", timeOnCPU);
	printf("Time on GPU: %.6f\n", timeOnGPU);

	// output


	// cleanup
	cleanup();

	return (0);
}

void cleanup(void)
{
	// code
	if (deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}

	if (deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceOutput = NULL;
	}

	if (deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceOutput = NULL;
	}

	if (gold)
	{
		free(gold);
		gold = NULL;
	}

	if (hostOutput)
	{
		cudaFree(hostOutput);
		hostOutput = NULL;
	}

	if (hostInput2)
	{
		cudaFree(hostInput2);
		hostOutput = NULL;
	}

	if (hostInput1)
	{
		cudaFree(hostInput1);
		hostOutput = NULL;
	}
}

void fillFloatArrayWithRandomNumbers(float* arr, int len)
{
	// code
	const float fScale = 1.0f / (float)RAND_MAX;
	for (int i = 0; i < len; i++)
	{
		arr[i] = fScale * rand();
	}
}

void vecAddCPU(const float* arr1, const float* arr2, float* out, int len)
{
	// code
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < len; i++)
	{
		out[i] = arr1[i] + arr2[i];
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

