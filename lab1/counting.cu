#include "counting.h"
#include "SyncedMemory.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include "Timer.h"

using namespace std;
__device__ __host__ int CeilDiv(int a, int b)
{
	return (a - 1) / b + 1;
}
__device__ __host__ int CeilAlign(int a, int b)
{
	return CeilDiv(a, b) * b;
}

__constant__ char* text;
__global__ void EstablishButtonTree(const char* text, int* pos, int dataSize) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < dataSize) {
		if (text[idx] == '\n') pos[idx] = 0;
		else pos[idx] = 1;
		//printf("i=%d,idx = %d, %d\n", dataSize, idx, pos[idx]);
	}
}
__global__ void EstablishLayerTree(int* tree, int * nextTree , int currentLayer, int layer_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < layer_size && idx%2 ==0 ) {
		if (tree[idx] & tree[idx + 1])
			nextTree[idx / 2] = tree[idx] + tree[idx + 1];
		else
			nextTree[idx / 2] = 0;

		//if(layer_size < 4000)printf("i=%d idx = %d,tree[%d][%d] = %d\n", layer_size, idx, currentLayer + 1, idx / 2, nextTree[idx / 2]);
	//printf("i=%d idx = %d,tree[%d][%d] = %d\n", layer_size, idx,currentLayer,idx, tree[idx]);
	}
}
__device__ int SetPosSeq(int currentTreeIdx, int currentIdx, int * treeLayerBeginSize[], int currentNum)
{

	if (currentIdx == 0){ 
		if (treeLayerBeginSize[currentTreeIdx][currentIdx]){ return 1; }
		else return 0;
	}

	if (treeLayerBeginSize[currentTreeIdx][currentIdx])
	{
		int i = 1;
		while (treeLayerBeginSize[currentTreeIdx][currentIdx-i])
		{

			i++;
			if (currentIdx - i < 0) break;
		}
		return i;
	}

	return 0;
}
__device__ int xyToIndex(int x, int y, int text_size)
{
	int result = 0;
	for (int i = 0; i<y; i++) {
		result += (int)(text_size * pow(0.5, i));
	}
	result += x;
	return result;
}

__device__ int SetPosRecur(int currentTreeIdx, int currentIdx, int * treeLayerBeginSize, bool topDown, int currentNum,int text_size)
{
	while (1) {


		if (!topDown && treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)] && currentTreeIdx == 0)
		{
			if (currentIdx % 2 == 0){//left ¡G + self
				currentNum += 1;
				if (!currentIdx) return currentNum;//boundary


				if (treeLayerBeginSize[xyToIndex(currentIdx - 1, currentTreeIdx, text_size)] != 0) {
					//
					currentIdx = currentIdx - 1;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, false, currentNum,text_size);
					//
				}
				return currentNum;

			}
			else if (currentIdx % 2 == 1) // right ¡G find parent
			{
				if (treeLayerBeginSize[xyToIndex(currentIdx / 2, currentTreeIdx + 1, text_size)] != 0) return SetPosRecur(currentTreeIdx + 1, currentIdx / 2, treeLayerBeginSize, false, currentNum, text_size);
				else {
					if (treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)]) {
						currentNum++;
					}
					return currentNum;
				}
			}

		}
		else if (!topDown && treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)] && currentTreeIdx != 0)
		{
			if (currentIdx % 2 == 0)
			{
				currentNum += treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)];
				if (!currentIdx) return currentNum;


				if (treeLayerBeginSize[xyToIndex(currentIdx - 1, currentTreeIdx, text_size)] == 0) { // topDown
					//
					currentIdx = currentIdx - 1;
					topDown = true;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx-1, treeLayerBeginSize, true, currentNum,text_size);
					//
				}
				//
				currentIdx = currentIdx - 1;
				topDown = false;
				continue;
				//return	SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, false, currentNum,text_size);
			}
			else if (currentIdx % 2 == 1)
			{
				if (treeLayerBeginSize[xyToIndex(currentIdx / 2, currentTreeIdx + 1, text_size)] != 0)
				{
					//
					currentTreeIdx += 1;
					currentIdx /= 2;
					topDown = false;
					continue;
					//return SetPosRecur(currentTreeIdx + 1, currentIdx / 2, treeLayerBeginSize, false, currentNum,text_size);
					//
				}
				else{
					currentNum += treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)];
					//
					currentIdx -= 1;
					topDown = true;

					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, true, currentNum,text_size);
					//
				}
			}
		}
		else if (topDown)
		{
			if (treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)] == 0 && currentIdx % 2 == 0){ // left topDown
				if (currentTreeIdx == 0) return currentNum;
				//
				currentTreeIdx -= 1;
				currentIdx = currentIdx * 2 + 1;
				topDown = true;
				//return SetPosRecur(currentTreeIdx - 1, currentIdx * 2 +1, treeLayerBeginSize, true, currentNum,text_size);
			}
			else if (currentIdx % 2 == 1){ // right
				if (treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)] != 0){
					currentNum += treeLayerBeginSize[xyToIndex(currentIdx, currentTreeIdx, text_size)];
					//
					currentIdx -= 1;
					topDown = true;
					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, true, currentNum,text_size);
				}
				else
				{
					if (currentTreeIdx == 0) return currentNum;
					//
					currentIdx = currentIdx * 2 + 1;
					currentTreeIdx -= 1;
					topDown = true;
					//return SetPosRecur(currentTreeIdx - 1, currentIdx * 2 +1, treeLayerBeginSize, true, currentNum,text_size);
				}
			}
		}

		return currentNum;
	}
}
__global__ void SetPosition(int  *tree, int text_size ,int * result)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//int idxTransNum = 0;

	
	//result[idx] = SetPosRecur(0, idx, tree, false, 0, text_size);
	int currentTreeIdx = 0;
	int currentIdx = idx;
	bool topDown = false;
	int currentNum = 0;
	while (1) {


		if (!topDown && tree[xyToIndex(currentIdx, currentTreeIdx, text_size)] && currentTreeIdx == 0)
		{
			if (currentIdx % 2 == 0){//left ¡G + self
				currentNum += 1;
				if (!currentIdx) break;//boundary


				if (tree[xyToIndex(currentIdx - 1, currentTreeIdx, text_size)] != 0) {
					//
					currentIdx = currentIdx - 1;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, false, currentNum,text_size);
					//
				}
				break;
			}
			else if (currentIdx % 2 == 1) // right ¡G find parent
			{
				if (tree[xyToIndex(currentIdx / 2, currentTreeIdx + 1, text_size)] != 0) {
					//
					currentIdx /= 2;
					currentTreeIdx += 1;
					topDown = false;
					continue;
					//return SetPosRecur(currentTreeIdx + 1, currentIdx / 2, treeLayerBeginSize, false, currentNum, text_size);
					//
				}
				else {
					if (tree[xyToIndex(currentIdx, currentTreeIdx, text_size)]) {
						currentNum++;
					}
					break;
				}
			}
		}
		else if (!topDown && tree[xyToIndex(currentIdx, currentTreeIdx, text_size)] && currentTreeIdx != 0)
		{
			if (currentIdx % 2 == 0)
			{
				currentNum += tree[xyToIndex(currentIdx, currentTreeIdx, text_size)];
				if (!currentIdx) break;
				if (tree[xyToIndex(currentIdx - 1, currentTreeIdx, text_size)] == 0) { // topDown
					//
					currentIdx = currentIdx - 1;
					topDown = true;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx-1, treeLayerBeginSize, true, currentNum,text_size);
					//
				}
				//
				currentIdx = currentIdx - 1;
				topDown = false;
				continue;
				//return	SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, false, currentNum,text_size);
			}
			else if (currentIdx % 2 == 1)
			{
				if (tree[xyToIndex(currentIdx / 2, currentTreeIdx + 1, text_size)] != 0)
				{
					//
					currentTreeIdx += 1;
					currentIdx /= 2;
					topDown = false;
					continue;
					//return SetPosRecur(currentTreeIdx + 1, currentIdx / 2, treeLayerBeginSize, false, currentNum,text_size);
					//
				}
				else{
					currentNum += tree[xyToIndex(currentIdx, currentTreeIdx, text_size)];
					//
					currentIdx -= 1;
					topDown = true;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, true, currentNum,text_size);
					//
				}
			}
		}
		else if (topDown)
		{
			if (tree[xyToIndex(currentIdx, currentTreeIdx, text_size)] == 0 && currentIdx % 2 == 0){ // left topDown
				if (currentTreeIdx == 0) break;
				//
				currentTreeIdx -= 1;
				currentIdx = currentIdx * 2 + 1;
				topDown = true;
				continue;
				//return SetPosRecur(currentTreeIdx - 1, currentIdx * 2 +1, treeLayerBeginSize, true, currentNum,text_size);
			}
			else if (currentIdx % 2 == 1){ // right
				if (tree[xyToIndex(currentIdx, currentTreeIdx, text_size)] != 0){
					currentNum += tree[xyToIndex(currentIdx, currentTreeIdx, text_size)];
					//
					currentIdx -= 1;
					topDown = true;
					continue;
					//return SetPosRecur(currentTreeIdx, currentIdx - 1, treeLayerBeginSize, true, currentNum,text_size);
				}
				else
				{
					if (currentTreeIdx == 0) break;
					//
					currentIdx = currentIdx * 2 + 1;
					currentTreeIdx -= 1;
					topDown = true;
					continue;
					//return SetPosRecur(currentTreeIdx - 1, currentIdx * 2 +1, treeLayerBeginSize, true, currentNum,text_size);
				}
			}
		}

		break;
	}

	result[idx] = currentNum;
	//printf("treeeeee[%d] =%d \n", idx, currentNum);
	/*
	if (idx == 0){
		if (treeLayerBeginSize[idx]){ idxTransNum=1; }
		else idxTransNum= 0;
	
	}else if (treeLayerBeginSize[idx])
	{
		int i = 1;
		while (treeLayerBeginSize[idx - i]>0)
		{
			i++;
			if (idx - i < 0) break;
		}
		idxTransNum = i;
	}
	else
	{
		idxTransNum = 0;
	}
	*/
	

}

void CountPosition(const char *text, int *pos, int text_size)
{

	int * treeArrayTmp = (int *)malloc(text_size*sizeof(int));
 	//int * treeArray;
	int * tree[10];
	int size = text_size;

	//cudaMalloc((void**)&treeArray, sizeof(int)*size);
	//cudaMemcpy(treeArray, treeArrayTmp, sizeof(int)*size, cudaMemcpyHostToDevice);

	int block_dim = text_size / 512 + 1;
	Timer buttonTimer;
	buttonTimer.Start();
	cudaMalloc((void**)&tree[0], sizeof(int)*size);
	cudaMemcpy(tree[0], treeArrayTmp, sizeof(int)*size, cudaMemcpyHostToDevice);
	
	EstablishButtonTree << <block_dim, 512 >> >(text, tree[0], text_size);
	cudaDeviceSynchronize();

	buttonTimer.Pause();
	//printf_timer(buttonTimer);
	//cudaMemcpy(treeArrayTmp, tree[0], sizeof(int)*text_size, cudaMemcpyDeviceToHost);


	int totalSize = text_size;
	int treeButtomSize = 512;

	int * treeTmp;
	Timer buttonTimer2;
	buttonTimer2.Start();

	for (int i = 1; i < 10; i++)
	{
		cudaMalloc((void**)&tree[i], sizeof(int)*totalSize);
		treeTmp = (int *)malloc(sizeof(int)*totalSize);
		cudaMemcpy(tree[i], treeTmp, sizeof(int)*totalSize, cudaMemcpyHostToDevice);
		block_dim = (totalSize) / treeButtomSize +1;

		EstablishLayerTree << <block_dim, treeButtomSize >> >(tree[i-1], tree[i], i-1, totalSize);
		cudaDeviceSynchronize(); 
		if (totalSize == 1) break;
		treeButtomSize /= 2;
		totalSize /= 2;
	}
	buttonTimer2.Pause();
	//printf_timer(buttonTimer2);


	// copy all tree
	totalSize = text_size;

	//printf("textsize = %d", totalSize);
	treeButtomSize = 512;
	int * treeBigArray = (int *)malloc(sizeof(int) * text_size * 2);
	int treeBigArrayIdx = 0;

	//
	for (int i = 0; i < 10; i++)
	{
		treeTmp = (int *)malloc(sizeof(int)*totalSize);
		cudaMemcpy(treeTmp, tree[i], sizeof(int)*totalSize, cudaMemcpyDeviceToHost);

		for (int j = 0; j < totalSize; j++)
		{
			treeBigArray[treeBigArrayIdx] = treeTmp[j];
			treeBigArrayIdx++;
		}
		totalSize /= 2;
		cudaFree(tree[i]);
	}

	//

	int * treeInOneD;
	cudaMalloc((void**)&treeInOneD, sizeof(int)*text_size * 2);

	cudaMemcpy(treeInOneD,treeBigArray,sizeof(int)* text_size * 2,cudaMemcpyHostToDevice);

	block_dim = text_size / 512 +1 ;
	//printf("dim  = %d",block_dim);
	int * treeResult;
	cudaMalloc((void**)&treeResult, sizeof(int)*text_size );
	Timer buttonTimer3;
	buttonTimer3.Start();
	
	SetPosition << <block_dim, 512 >> >(treeInOneD, text_size, treeResult);
	
	cudaDeviceSynchronize();


	buttonTimer3.Pause();
	//printf_timer(buttonTimer3);
	
	//treeTmp = (int *)malloc(sizeof(int)*text_size *2);

	//cudaMemcpy(treeTmp, treeInOneD, sizeof(int)*text_size * 2, cudaMemcpyDeviceToHost);

	//int * treett = (int*)malloc(sizeof(int)*text_size);
	//cudaMemcpy(treett, tree[0], sizeof(int)*text_size, cudaMemcpyDeviceToHost);

	//char * tmpText = (char *)malloc(sizeof(char)* text_size);
	//cudaMemcpy(tmpText, text, sizeof(char)*text_size, cudaMemcpyDeviceToHost);

	int * treeResultTmp = (int*)malloc(sizeof(int)* text_size);
	cudaMemcpy(treeResultTmp, treeResult, sizeof(int)*text_size, cudaMemcpyDeviceToHost);

	
	for (int j = 0; j < 100; j++)
	{
		//printf("treetmp[%d] =%d\n", j, treeResultTmp[j]);
	}

	cudaMemcpy(pos, treeResult, sizeof(char)*text_size * 4, cudaMemcpyHostToDevice);
}
template<int N>
class compare {
public:
	__device__ bool operator () (int x) { return x == N; }
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead = 0;
	cudaMalloc((void**)&buffer, sizeof(int)*text_size * 2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer + text_size);

	// TODO
	auto head_end_d =
		thrust::copy_if(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(text_size),
		pos_d,
		head_d,
		compare<1>()
		);
	nhead = head_end_d - head_d;
	cudaFree(buffer);
	return nhead;
}

__global__ void P3Extra(char *input_gpu, int fsize)
{
	int idx = blockIdx.x * blockDim.x * threadIdx.x;
	if ((idx < fsize) && (input_gpu[idx] != '\n')) {

		//input_gpu[idx] = input_gpu[fsize-1-idx];

		if (idx % 2 == 0 && (input_gpu[idx] != ' '))
		{
			if ((input_gpu[idx] != ' ') && (input_gpu[idx + 1] != ' ') && (input_gpu[idx] != '.') && (input_gpu[idx + 1] != '.'))
			{
				// swap
				char tmp = input_gpu[idx];
				input_gpu[idx] = input_gpu[idx + 1];
				input_gpu[idx + 1] = tmp;
				//}
				if (input_gpu[idx - 1] == ' ' || !input_gpu[idx - 1])
				{
					if (input_gpu[idx] >= 'a' && input_gpu[idx] <= 'z')
					{
						input_gpu[idx] -= 32;
					}
					if (input_gpu[idx + 1] >= 'A' && input_gpu[idx + 1] <= 'Z')
					{
						input_gpu[idx + 1] += 32;
					}
				}
				else
				{
					if (input_gpu[idx] >= 'A' && input_gpu[idx] <= 'Z')
					{
						input_gpu[idx] += 32;
					}
					if (input_gpu[idx + 1] >= 'A' && input_gpu[idx + 1] <= 'Z')
					{
						input_gpu[idx + 1] += 32;
					}
				}
			}
			else if (input_gpu[idx + 1] == ' '&&input_gpu[idx + 1] == '.')
			{
				if (input_gpu[idx] >= 'A' && input_gpu[idx] <= 'Z')
				{
					input_gpu[idx] += 32;
				}
			}
		}
		else if (idx % 2 == 1)
		{

		}
	}
}
void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	int blockdim = text_size / 512 + 1;

	char *input_gpu = (char*)malloc(sizeof(char)*text_size);


	P3Extra << < blockdim, 512 >> > (text,text_size);

	

}
