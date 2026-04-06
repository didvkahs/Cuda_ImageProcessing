#include <cstdint>

#include "BmpBuffer.h"
#include "CudaProcessor.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>



#define SAFE_RELEASE(p) {if(p != nullptr){ cudaFree(p); p = nullptr;}}

constexpr int CU_IMG_W = 1024;
constexpr int CU_IMG_H = 1024;
constexpr uint32_t MAX_PIXEL_VAL = 256;

constexpr int BLUR_RADIUS = 2;
constexpr int BLOCK = 16;

constexpr float THRESHOLD = 0.15f;

inline bool CudaCheck(cudaError_t err);
inline bool CudaKernelCheck(void);

__constant__ uint32_t KC_WARP_SIZE = 32;
__constant__ uint32_t KC_MASK = 0xffffffff; // indicates that all 32 threads in the warp participate

__global__ void histogram(uint32_t* histogramVal, uint8_t* buf, size_t m_pitch, uint32_t width, uint32_t height);
__global__ void horizontalBlur(uint8_t* dstBuf, const uint8_t* srcBuf, size_t pitch, int width, int height);
__global__ void verticalBlur(uint8_t* dstBuf, const uint8_t* srcBuf, size_t pitch, int width, int height);
__global__ void replaceWithHisto(uint32_t* histo, uint8_t* ioBuf, size_t pitch, uint32_t width, uint32_t height);
__global__ void laplacian(uint8_t* dstBuf, uint8_t* srcBuf, size_t pitch, uint32_t width, uint32_t height);


CudaProcessor::CudaProcessor(void) {}
CudaProcessor::~CudaProcessor(void) {}

bool CudaProcessor::Initialize(void)
{
	if (!CudaCheck(cudaMalloc(&m_dHisto, MAX_PIXEL_VAL * sizeof(uint32_t))))
	{
		goto LB_FAILED_ALLOC_HISTOGRAM;
	}

	if (!CudaCheck(cudaMallocPitch(&m_dr, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		goto LB_FAILED_ALLOC_DEVMEM1;
	}
	if (!CudaCheck(cudaMallocPitch(&m_dg, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		cudaFree(m_dr);
		goto LB_FAILED_ALLOC_DEVMEM1;
	}
	if (!CudaCheck(cudaMallocPitch(&m_db, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		cudaFree(m_dr);
		cudaFree(m_dg);
		goto LB_FAILED_ALLOC_DEVMEM1;
	}

	if (!CudaCheck(cudaMallocPitch(&m_dr2, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		goto LB_FAILED_ALLOC_DEVMEM2;
	}
	if (!CudaCheck(cudaMallocPitch(&m_dg2, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		cudaFree(m_dr2);
		goto LB_FAILED_ALLOC_DEVMEM2;
	}
	if (!CudaCheck(cudaMallocPitch(&m_db2, &m_pitch, static_cast<size_t>(CU_IMG_W), static_cast<size_t>(CU_IMG_H))))
	{
		cudaFree(m_dr2);
		cudaFree(m_dg2);
		goto LB_FAILED_ALLOC_DEVMEM2;
	}


	return true;

LB_FAILED_ALLOC_DEVMEM2:
	cudaFree(m_dr);
	cudaFree(m_dg);
	cudaFree(m_db);

LB_FAILED_ALLOC_DEVMEM1:
	cudaFree(m_dHisto);

LB_FAILED_ALLOC_HISTOGRAM:

	return false;
}


bool CudaProcessor::ComputeHistogram(RAWImageBuf_s*& inBuf, uint32_t*& outHisto)
{
	resetDevBufs();

	m_imgW = inBuf->imgW;
	m_imgH = inBuf->imgH;

	if (!CudaCheck(cudaMemcpy2D(m_dr, m_pitch, inBuf->r, sizeof(uint8_t) * m_imgW, m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}

	{
		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);
		histogram << <grid, block >> > (m_dHisto, m_dr, m_pitch, m_imgW, m_imgH);
		if (!CudaKernelCheck()) { return false; }

		if(!CudaCheck(cudaMemcpy(outHisto, m_dHisto, MAX_PIXEL_VAL * sizeof(uint32_t), cudaMemcpyDeviceToHost)))
		{
			fprintf(stderr, "compute Histogram cudaMemcpy failed DTOH\n");
			goto LB_FAILED_MEMCPY_DTOH;
		}
	}

	return true;

	LB_FAILED_MEMCPY_DTOH:
	LB_FAILED_MEMCPY_HTOD:

	return false;
}

bool CudaProcessor::ApplyBlur(RAWImageBuf_s*& ioBuf)
{
	resetDevBufs();

	m_imgW = ioBuf->imgW;
	m_imgH = ioBuf->imgH;

	
	if (!CudaCheck(cudaMemcpy2D(m_dr, m_pitch, ioBuf->r, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_dg, m_pitch, ioBuf->g, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_db, m_pitch, ioBuf->b, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}

	{
		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);
		int shMemSizeH = (block.x + 2 * BLUR_RADIUS) * block.y * sizeof(uint8_t);
		int shMemSizeV = (block.y * (block.y + 2 * BLUR_RADIUS)) * sizeof(uint8_t);

		horizontalBlur <<<grid, block, shMemSizeH >>> (m_dr2, m_dr, m_pitch, m_imgW, m_imgH);
		horizontalBlur <<<grid, block, shMemSizeH >>> (m_dg2, m_dg, m_pitch, m_imgW, m_imgH);
		horizontalBlur <<<grid, block, shMemSizeH >>> (m_db2, m_db, m_pitch, m_imgW, m_imgH);

		verticalBlur <<<grid, block, shMemSizeV >>> (m_dr, m_dr2, m_pitch, m_imgW, m_imgH);
		verticalBlur <<<grid, block, shMemSizeV >>> (m_dg, m_dg2, m_pitch, m_imgW, m_imgH);
		verticalBlur <<<grid, block, shMemSizeV >>> (m_db, m_db2, m_pitch, m_imgW, m_imgH);

		if (!CudaKernelCheck())
		{
			fprintf(stderr, "Apply blur failed\n");
			goto LB_FAILED_KERNEL;
		}
	}

	if (!CudaCheck(cudaMemcpy2D(ioBuf->r, m_imgW, m_dr, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->g, (m_imgW), m_dg, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->b, (m_imgW), m_db, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}


	return true;

LB_FAILED_MEMCPY_DTOH:

LB_FAILED_KERNEL:

LB_FAILED_MEMCPY_HTOD:

	return false;
}

bool CudaProcessor::ApplyEqualize(RAWImageBuf_s*& ioBuf, uint32_t*& ioHisto)
{
	resetDevBufs();

	m_imgW = ioBuf->imgW;
	m_imgH = ioBuf->imgH;

	if (!CudaCheck(cudaMemcpy2D(m_dr, m_pitch, ioBuf->r, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_dg, m_pitch, ioBuf->g, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_db, m_pitch, ioBuf->b, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}

	{
		const uint32_t HISTOGRAM_SIZE = MAX_PIXEL_VAL * sizeof(uint32_t);
		const uint32_t DENOMINATOR = m_imgW * m_imgH - 1;

		float cdf = 0.0f; // cumulate count

		for (uint32_t i = 0; i < MAX_PIXEL_VAL; ++i)
		{
			cdf += ioHisto[i];
			float h = (cdf / DENOMINATOR) * 255.0f;

			ioHisto[i] = (uint8_t)(h + 0.5f);
		}

		cudaMemcpy(m_dHisto, ioHisto, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithHisto << <grid, block >> > (m_dHisto, m_dr, m_pitch, m_imgW, m_imgH);
		replaceWithHisto << <grid, block >> > (m_dHisto, m_dg, m_pitch, m_imgW, m_imgH);
		replaceWithHisto << <grid, block >> > (m_dHisto, m_db, m_pitch, m_imgW, m_imgH);

		if (!CudaKernelCheck())
		{
			fprintf(stderr, "ApplyEqualize kernel failed\n");
			goto LB_FAILED_KERNEL;
		}

	}

	if (!CudaCheck(cudaMemcpy2D(ioBuf->r, m_imgW * sizeof(uint8_t), m_dr, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->g, m_imgW * sizeof(uint8_t), m_dg, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->b, m_imgW * sizeof(uint8_t), m_db, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}

	return true;

LB_FAILED_MEMCPY_DTOH:

LB_FAILED_KERNEL:

LB_FAILED_MEMCPY_HTOD:

	return false;
}


bool CudaProcessor::ApplyColorMap(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto)
{
	resetDevBufs();

	const uint32_t HISTOGRAM_SIZE = MAX_PIXEL_VAL * sizeof(uint32_t);


	m_imgW = ioBuf->imgW;
	m_imgH = ioBuf->imgH;

	uint32_t* histoR = nullptr;
	uint32_t* histoG = nullptr;
	uint32_t* histoB = nullptr;

	if (!CudaCheck(cudaMemcpy2D(m_dr, m_pitch, ioBuf->r, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_dg, m_pitch, ioBuf->g, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_db, m_pitch, ioBuf->b, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}

	if (!CudaCheck(cudaMalloc(&histoR, HISTOGRAM_SIZE)))
	{
		goto LB_FAILED_MALLOC_HISTOGRAM;
	}
	if (!CudaCheck(cudaMalloc(&histoG, HISTOGRAM_SIZE)))
	{
		goto LB_FAILED_MALLOC_HISTOGRAM;
	}
	if (!CudaCheck(cudaMalloc(&histoB, HISTOGRAM_SIZE)))
	{
		goto LB_FAILED_MALLOC_HISTOGRAM;
	}


	{

		isoDataClustering(inHisto);

		Cluster_s* cluster = m_clusters;

		uint32_t* hHistoR = new uint32_t[MAX_PIXEL_VAL];
		uint32_t* hHistoG = new uint32_t[MAX_PIXEL_VAL];
		uint32_t* hHistoB = new uint32_t[MAX_PIXEL_VAL];

		while (cluster != nullptr)
		{
			uint32_t startIDX = cluster->startIDX;
			uint32_t endIDX = startIDX + cluster->length;

			uint32_t r = rand() % MAX_PIXEL_VAL;
			uint32_t g = rand() % MAX_PIXEL_VAL;
			uint32_t b = rand() % MAX_PIXEL_VAL;

			for (uint32_t i = startIDX; i < endIDX; ++i)
			{
				hHistoR[i] = r;
				hHistoG[i] = g;
				hHistoB[i] = b;
			}

			cluster = cluster->next;
		}

		cudaMemcpy(histoR, hHistoR, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(histoG, hHistoG, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(histoB, hHistoB, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);


		delete[] hHistoR;
		delete[] hHistoG;
		delete[] hHistoB;

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithHisto <<<grid, block>>>(histoR, m_dr, m_pitch, m_imgW, m_imgH);
		replaceWithHisto <<<grid, block>>>(histoG, m_dg, m_pitch, m_imgW, m_imgH);
		replaceWithHisto <<<grid, block>>>(histoB, m_db, m_pitch, m_imgW, m_imgH);

		if (!CudaKernelCheck())
		{
			fprintf(stderr, "ColorMap kernel failed\n");
			goto LB_FAILED_KERNEL;
		}
	}

	if (!CudaCheck(cudaMemcpy2D(ioBuf->r, m_imgW * sizeof(uint8_t), m_dr, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->g, m_imgW * sizeof(uint8_t), m_dg, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->b, m_imgW * sizeof(uint8_t), m_db, m_pitch, m_imgW * sizeof(uint8_t), m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}


	return true;

LB_FAILED_MEMCPY_DTOH:

LB_FAILED_KERNEL:

	cudaFree(histoR);
	cudaFree(histoG);
	cudaFree(histoB);

LB_FAILED_MALLOC_HISTOGRAM:

LB_FAILED_MEMCPY_HTOD:

	return false;
}


bool CudaProcessor::ApplyLaplacian(RAWImageBuf_s*& ioBuf)
{
	resetDevBufs();

	m_imgW = ioBuf->imgW;
	m_imgH = ioBuf->imgH;

	if (!CudaCheck(cudaMemcpy2D(m_dr, m_pitch, ioBuf->r, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_dg, m_pitch, ioBuf->g, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}
	if (!CudaCheck(cudaMemcpy2D(m_db, m_pitch, ioBuf->b, sizeof(uint8_t) * m_imgW, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_MEMCPY_HTOD;
	}

	{
		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		laplacian<<<grid, block>>>(m_dr2, m_dr, m_pitch, m_imgW, m_imgH);
		laplacian<<<grid, block>>>(m_dg2, m_dg, m_pitch, m_imgW, m_imgH);
		laplacian<<<grid, block>>>(m_db2, m_db, m_pitch, m_imgW, m_imgH);

		if (!CudaKernelCheck())
		{
			fprintf(stderr, "apply laplacian kernel failed\n");
			goto LB_FAILED_KERNEL;
		}
	}

	
	if (!CudaCheck(cudaMemcpy2D(ioBuf->r, sizeof(uint8_t) * m_imgW, m_dr2, m_pitch, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->g, sizeof(uint8_t) * m_imgW, m_dg2, m_pitch, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->b, sizeof(uint8_t) * m_imgW, m_db2, m_pitch, sizeof(uint8_t) * m_imgW, m_imgH, cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}



	return true;

LB_FAILED_MEMCPY_DTOH:

LB_FAILED_KERNEL:

LB_FAILED_MEMCPY_HTOD:

	return false;
}




void CudaProcessor::CloseCudaHandles(void)
{
	SAFE_RELEASE(m_dHisto)
	SAFE_RELEASE(m_dr)
	SAFE_RELEASE(m_dg)
	SAFE_RELEASE(m_db)
		
	SAFE_RELEASE(m_dr2)
	SAFE_RELEASE(m_dg2)
	SAFE_RELEASE(m_db2)

	while (m_clusters != nullptr)
	{
		Cluster_s* next = m_clusters->next;
		delete m_clusters;
		m_clusters = next;
	}
	m_clusters = nullptr;
}










void CudaProcessor::resetDevBufs(void)
{
	static const int DBUF_SIZE = CU_IMG_H * CU_IMG_W;
	static const int HISTOGRAM_SIZE = MAX_PIXEL_VAL * sizeof(int);

	m_imgW = 0;
	m_imgH = 0;

	cudaMemset(m_dr, 0, DBUF_SIZE);
	cudaMemset(m_dg, 0, DBUF_SIZE);
	cudaMemset(m_db, 0, DBUF_SIZE);

	cudaMemset(m_dr2, 0, DBUF_SIZE);
	cudaMemset(m_dg2, 0, DBUF_SIZE);
	cudaMemset(m_db2, 0, DBUF_SIZE);

	cudaMemset(m_dHisto, 0, HISTOGRAM_SIZE);

	if (!CudaKernelCheck())
	{
		fprintf(stderr, "cudaMemset failed \n");
#ifdef _DEBUG
		__debugbreak();
#endif
		exit(1);
	}
}

void CudaProcessor::isoDataClustering(uint32_t*& histo)
{
	m_clusters = new Cluster_s;

	for (uint32_t i = 0; i < MAX_PIXEL_VAL; ++i)
	{
		m_clusters->cv = calCV(histo, m_clusters->startIDX, m_clusters->length);

		if (m_clusters->cv > THRESHOLD)
		{
			// TODO : divide clusters to 2 different cluster

			Cluster_s* newNode = new Cluster_s;
			m_clusters->length -= 1;
			
			newNode->startIDX = m_clusters->startIDX + m_clusters->length;
			newNode->next = m_clusters;

			m_clusters = newNode;
		}

		m_clusters->length += 1;
	}
}


inline float CudaProcessor::calCV(uint32_t* histo, uint32_t startIDX, uint32_t length)
{
	float mean = 0.0f;
	float variance = 0.0f;
	float stddev = 0.0f; // standard deviation

	uint32_t sum = 0;
	uint32_t totalCount = 0;
	float sumSqDiff = 0.0f;

	for (uint32_t i = startIDX; i < startIDX + length; ++i)
	{
		sum += histo[i] * i;
		totalCount += histo[i];
	}

	if (sum == 0)
	{
		return 0;
	}

	mean = static_cast<float>(sum) / static_cast<float>(totalCount);

	for (uint32_t i = startIDX; i < startIDX + length; ++i)
	{
		float diff = (float)i - mean;
		sumSqDiff += (diff * diff) * histo[i];
	}

	variance = sumSqDiff / (float)totalCount;
	stddev = std::sqrt(variance);

	float cv = stddev / mean;

	return cv;
}









__global__ void horizontalBlur(uint8_t* dstBuf, const uint8_t* srcBuf, size_t pitch, int width, int height)
{
	extern __shared__ uint8_t sharedMem[];

	const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int shPitch = blockDim.x + 2 * BLUR_RADIUS;

	if (srcX >= width || srcY >= height) return;

	for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i)
	{
		int nsrcX = min(max(srcX + i, 0), width - 1);
		int sharedX = tx + i + BLUR_RADIUS;
		sharedMem[sharedX + shPitch * ty] = srcBuf[nsrcX + pitch * srcY];
	}

	__syncthreads();

	uint32_t sum = 0;
	for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i)
	{
		int sharedX = tx + i + BLUR_RADIUS;
		sum += sharedMem[sharedX + shPitch * ty];
	}

	uint8_t val = static_cast<uint8_t>(min(max(sum / float(2 * BLUR_RADIUS + 1), 0.0f), 255.0f));

	dstBuf[srcX + pitch * srcY] = val;
}

__global__ void verticalBlur(uint8_t* dstBuf, const uint8_t* srcBuf, size_t pitch, int width, int height)
{
	extern __shared__ uint8_t sharedMem[];

	const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int shPitch = blockDim.x;

	if (srcX >= width || srcY >= height) return;

	for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i)
	{
		int nsrcY = min(max(srcY + i, 0), width - 1);
		int sharedY = ty + i + BLUR_RADIUS;
		sharedMem[tx + shPitch * sharedY] = srcBuf[srcX + pitch * nsrcY];
	}

	__syncthreads();

	uint32_t sum = 0;
	for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i)
	{
		int sharedY = ty + i + BLUR_RADIUS;
		sum += sharedMem[tx + shPitch * sharedY];
	}

	uint8_t val = static_cast<uint8_t>(min(max(sum / float(2 * BLUR_RADIUS + 1), 0.0f), 255.0f));

	dstBuf[srcX + pitch * srcY] = val;
}

__global__ void histogram(uint32_t* histogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

	uint8_t pixelVal = buf[srcX + pitch * srcY];

	atomicAdd(&histogramVal[pixelVal], 1);
}

__global__ void replaceWithHisto(uint32_t* histogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

	uint8_t pixelVal = buf[srcX + pitch * srcY];
	uint8_t equalizedVal = histogramVal[pixelVal];

	buf[srcX + pitch * srcY] = equalizedVal;
}

__global__ void laplacian(uint8_t* dstBuf, uint8_t* srcBuf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t tx = threadIdx.x;
	const uint32_t ty = threadIdx.y;

	if (srcX >= width || srcY >= height) return;

	uint8_t center = srcBuf[srcX + pitch * srcY];

	uint8_t lp_shfl = __shfl_up_sync(KC_MASK, center, 1);
	uint8_t rp_shfl = __shfl_down_sync(KC_MASK, center, 1);

	uint8_t lp, rp;

	if (tx == 0)
		lp = srcBuf[max((int)srcX - 1, 0) + pitch * srcY];
	else
		lp = lp_shfl;

	if (tx == BLOCK - 1 || srcX == width - 1)
		rp = srcBuf[min((int)srcX + 1, (int)width - 1) + pitch * srcY];
	else
		rp = rp_shfl;

	uint32_t upY = min(max(srcY - 1, 0), height - 1);
	uint32_t downY = min(max(srcY + 1, 0), height + 1);

	uint8_t up = srcBuf[srcX + pitch * upY];
	uint8_t dp = srcBuf[srcX + pitch * downY];

	int res = (int)lp + (int)rp + (int)up + (int)dp - 4 * (int)center;
	dstBuf[srcX + pitch * srcY] = (uint8_t)max(0, min(255, res));
}




inline bool CudaCheck(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error : " << cudaGetErrorString(err) << "at line " << __LINE__ << "\n";
		return false;
	}
	return true;
}

inline bool CudaKernelCheck(void)
{
	return CudaCheck(cudaGetLastError());
}