#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"
#include <cuda.h>
#include <string>

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

// pair word invert
__global__ void SomeTransform(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx < fsize) && (input_gpu[idx] != '\n')) {
		if (idx%2==0 && (input_gpu[idx]!=' '))
		{
			if ((input_gpu[idx] != ' ') && (input_gpu[idx + 1] != ' ') && (input_gpu[idx] != '.') && (input_gpu[idx + 1] != '.'))
			{
					char tmp = input_gpu[idx];
					input_gpu[idx] = input_gpu[idx + 1];
					input_gpu[idx + 1] = tmp;
				
					if (input_gpu[idx - 1] == ' ' || !input_gpu[idx - 1])
					{
						if (input_gpu[idx] >= 'a' && input_gpu[idx] <='z')
						{
							input_gpu[idx] -= 32;
						}
						if (input_gpu[idx + 1] >= 'A' && input_gpu[idx + 1] <= 'Z')
						{
							input_gpu[idx+1] += 32;
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
		else if (idx%2 ==1 )
		{

		}
	}
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if ( !fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);
	char *input_gpu = text_smem.get_gpu_rw();
	SomeTransform<<<fsize/32, 32>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());

	return 0;
	
}
