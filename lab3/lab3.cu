#include "lab3.h"
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x; //target.x
	const int yt = blockIdx.y * blockDim.y + threadIdx.y; //target.y

	const int currentNum = wt*yt + xt;

	if (yt < ht && xt < wt && mask[currentNum] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int currentBG = wb*yb + xb;

		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[currentBG * 3 + 0] = target[currentNum * 3 + 0];
			output[currentBG * 3 + 1] = target[currentNum * 3 + 1];
			output[currentBG * 3 + 2] = target[currentNum * 3 + 2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int xt = blockDim.x * blockIdx.x + threadIdx.x;
	const int yt = blockDim.y * blockIdx.y + threadIdx.y;
	const int currentNum = wt * yt + xt;

	float t_sum, b_sum;

	if (yt < ht && xt < wt) {
		const int yb = oy + yt, xb = ox + xt;
		const int currentBG = wb*yb + xb;

		for (int i = 0; i < 3; i++){
			t_sum = 0;
			b_sum = 0;

			if (xt - 1 >= 0){
				b_sum += mask[(currentNum - 1)] < 127.0 ? background[(currentBG - 1) * 3 + i] : 0;
				t_sum += target[(currentNum - 1) * 3 + i];
			}
			else{
				b_sum += background[(currentBG - 1) * 3 + i];
				t_sum += target[(currentNum)* 3 + i];
			}

			if (xt + 1 < wt){
				b_sum += mask[(currentNum + 1)] < 127.0 ? background[(currentBG + 1) * 3 + i] : 0;
				t_sum += target[(currentNum + 1) * 3 + i];
			}
			else{
				b_sum += background[(currentBG + 1) * 3 + i];
				t_sum += target[(currentNum)* 3 + i];
			}

			if (yt - 1 >= 0){
				b_sum += mask[(currentNum - wt)] < 127.0 ? background[(currentBG - wb) * 3 + i] : 0;
				t_sum += target[(currentNum - wt) * 3 + i];
			}
			else{
				b_sum += background[(currentBG - wb) * 3 + i];
				t_sum += target[(currentNum)* 3 + i];
			}

			if (yt + 1 < ht){
				b_sum += mask[(currentNum + wt)] < 127.0 ? background[(currentBG + wb) * 3 + i] : 0;
				t_sum += target[(currentNum + wt) * 3 + i];
			}
			else{
				b_sum += background[(currentBG + wb) * 3 + i];
				t_sum += target[(currentNum)* 3 + i];
			}

			fixed[currentNum * 3 + i] = 4.0*target[currentNum * 3 + i] - t_sum + b_sum;
		}
	}
}

__global__ void PoissonImageCloningInteration(
	const float *fixed,
	const float *mask,
	const float *target,
	float *output,
	const int wt,
	const int ht
	)
{

	const int xt = blockIdx.x * blockDim.x + threadIdx.x; //target.x
	const int yt = blockIdx.y * blockDim.y + threadIdx.y; //target.y
	const int currentNum = wt * yt + xt;

	float neibor_sum = 0;
	if (yt < ht && xt < wt && mask[currentNum] > 127.0f){
		for (int i = 0; i < 3; i++){
			neibor_sum = 0;

			// left
			if (xt - 1 >= 0 && mask[(currentNum - 1)] > 127.0f){
				neibor_sum += target[(currentNum - 1) * 3 + i];
			}
			// right
			if (xt + 1 < wt && mask[(currentNum + 1)] > 127.0f){
				neibor_sum += target[(currentNum + 1) * 3 + i];
			}
			// up
			if (yt - 1 >= 0 && mask[(currentNum - wt)] > 127.0f){
				neibor_sum += target[(currentNum - wt) * 3 + i];
			}
			// down
			if (yt + 1 < ht && mask[(currentNum + wt)] > 127.0f){
				neibor_sum += target[(currentNum + wt) * 3 + i];
			}

			/*if (idx == 0)
			{
				output[currentNum * 3 + i] = (fixed[currentNum * 3 + i] + neibor_sum) / 4;

			}
			if (idx == 1)
			{*/
				float w = 1.414;
				output[currentNum * 3 + i] = w * (fixed[currentNum * 3 + i] + neibor_sum) / 4 + (1.0 - w) *output[currentNum * 3 + i];
			//}



		}
	}
}
void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{

	//set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3 * wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3 * wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3 * wt*ht*sizeof(float));

	//initialize the iteration
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CalculateFixed << <gdim, bdim >> >(background, target, mask, fixed, wb, hb, wt, ht, oy, ox);

	cudaMemcpy(buf1, target, sizeof(float) * 3 * wt * ht, cudaMemcpyDeviceToDevice);

	//iterate
	int iteratorNum = 5000;
	//if (idx == 0)  iteratorNum = 10000;
	//if (idx == 1)  iteratorNum = 5000;
	for (int i = 0; i < iteratorNum; ++i){
		PoissonImageCloningInteration << <gdim, bdim >> >(fixed, mask, buf1, buf2, wt, ht);
		PoissonImageCloningInteration << <gdim, bdim >> >(fixed, mask, buf2, buf1, wt, ht);
	}
	//copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	SimpleClone << <gdim, bdim >> >(background, buf1, mask, output, wb, hb, wt, ht, oy, ox);

	//clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}
