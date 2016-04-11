#include "lab2.h"

// Macro functions - YUV and RGB converting.
#define clip(x) ((x) > 255 ? 255 : (x) < 0 ? 0 : (int)x)
#define RGBtoY(R, G, B) clip( ( 0.299 * R) + ( 0.587 * G) + ( 0.114 * B)      )
#define RGBtoU(R, G, B) clip( (-0.169 * R) + (-0.331 * G) + ( 0.500 * B) + 128)
#define RGBtoV(R, G, B) clip( ( 0.500 * R) + (-0.419 * G) + (-0.081 * B) + 128)

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 960;

struct Vector3D
{
	unsigned char x, y, z;
	Vector3D(unsigned char x, unsigned char y, unsigned char z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
};

Lab2VideoInfo tmpInfo;
struct Lab2VideoGenerator::Impl {
	int t = 0;
};



Lab2VideoGenerator::Lab2VideoGenerator() : impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

__global__ void PCG2()
{


}
__device__ int* PCGRecurOne(int x, int y , int t , int part)
{

	//printf("x = %d , y = %d ,cosx = %f\n ",x,y,cosf(x));
	int windowDis = H*H + W*W;
	int dis = x*x + y*y;
	float disMod = (float)dis / (float)windowDis;
	//printf("dixmod =%f ,%d \n", (255 * disMod), (int)(255 * disMod));
	if (part == 1)
	{
		int tmpt = 
		t < 10 ? t = 10 : t;

		float func = cosf(y / t) - cosf(x / t);
		if (func < 0.1 && func> -0.1)
		{
			int RGB[3] = { 0, 255,  255*disMod};
			//printf("1RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}

		else
		{
			int RGB[3] = { -1, -1, -1 };
			//printf("3RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}
	}
	else if (part == 2){
		//float func = cosf(y + t) + cosf(x +t);

		float tmpt = t / NFRAME;
		t > NFRAME / 2 ? tmpt = 1 : tmpt = 6;
		float func = sinf(x+t) + sinf(y+t);
		if (func < 0.1 && func> -0.1)
		{
			int RGB[3] = { 100 - 100 * tmpt , 30   , 50 -50 * tmpt };
			//printf("1RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}

		else
		{
			int RGB[3] = { -1, -1, -1 };
			//printf("3RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}
	}
	else if (part == 3)
	{
		int tMod = t%W;
		
		float func = x*x + y*y - (2 * t*tMod);
		if (func <1000 * tMod + t && func > -1000 * tMod)
		{
		//	if (t > 30)printf("func =  %d,%d,%d \n", x, y, t);

			int RGB[3] = { 255 - 20*disMod , 189 - 80*disMod, 52 + 40*disMod };
			//printf("1RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}
		else
		{
			int RGB[3] = { -1, -1, -1 };
			//printf("3RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
			return RGB;
		}
	}
	int RGB[3] = {-1,-1,-1};
	//printf("4RGB = %d , %d , %d\n", RGB[0], RGB[1], RGB[2]);
	return RGB;
}

__global__ void PCG(Lab2VideoInfo &info, uint8_t * yuv, int tt)
{
	int t = tt;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int tPCG = t;
	//PCG2 << < 1, 1 >> > ();
	int width = idx % W;
	int height = idx / W;

	int uvWidth = width / 2;
	int uvHeight = height / 2;

	int uvIdx = uvWidth + uvHeight * (W / 2);
	int modT = t < 230 ? 240 - t : 0;
	int RGB[3] = { -1, -1, -1 };
	//int RGB[3] = { 255, 0, 0 };
	int *RGBtmp;
	for (int i = 1; i <=3; i++)
	{
		
		if (i==1) RGBtmp = PCGRecurOne(width, height, modT, i);
		
		if (RGBtmp[0] == -1 || RGBtmp[1] == -1 || RGBtmp[2] == -1) continue;
		else
		{
			if (i == 2){ int * RGBTmp2 = PCGRecurOne(width, height, tPCG, i); 
			if (RGBTmp2[0] != -1)
			{
				RGBtmp[0] = RGBTmp2[0];
				RGBtmp[1] = RGBTmp2[1];
				RGBtmp[2] = RGBTmp2[2];
			}
			}
			if (i == 3){
				int * RGBTmp2 = PCGRecurOne(width, height, tPCG, i);
				if (RGBTmp2[0] != -1)
				{
					RGBtmp[0] = RGBTmp2[0];
					RGBtmp[1] = RGBTmp2[1];
					RGBtmp[2] = RGBTmp2[2];
				}
			}

		}
		for (int j = 0; j < 3; j++)
		{
			RGB[j] = RGBtmp[j];
		}
		
	}
	
	if (RGB[0] == -1 || RGB[1] == -1 || RGB[2] == -1) return;

	yuv[idx] = (int)RGBtoY(RGB[0], RGB[1], RGB[2]);
	yuv[W*H + uvIdx] = (int)RGBtoU(RGB[0], RGB[1], RGB[2]);
	yuv[W*H + W*H / 4 + uvIdx] =(int) RGBtoV(RGB[0], RGB[1], RGB[2]);

}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;


	tmpInfo = info;
};


void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	
	cudaMemset(yuv,  0, W*H);
	cudaMemset(yuv + W*H, 128, W*H / 2);

	int block_dim = H*W / W;
	int t= impl->t;
	//cudaMemcpy(&t, &impl->t, sizeof(int), cudaMemcpyHostToDevice);
	PCG << <block_dim, W >> >(tmpInfo, yuv, t);

	++(impl->t);
}