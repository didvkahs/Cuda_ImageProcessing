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
__constant__ uint32_t KC_QUATER = 4;

__global__ void avgBlur(uint8_t* dstBuf, uint8_t* srcBuf, size_t pich, uint32_t width, uint32_t height);
__global__ void laplacian(uint8_t* dstBuf, uint8_t* srcBuf, size_t pitch, uint32_t width, uint32_t height);
__global__ void replaceWithLUT(uint8_t* LUT, uint8_t* ioBuf, size_t pitch, uint32_t width, uint32_t height);
__global__ void replaceWithRGBLUT(uint8_t* rgbLUT, uint8_t* r, uint8_t* g, uint8_t* b, size_t pitch, uint32_t width, uint32_t height);

CudaProcessor::CudaProcessor(void) {}
CudaProcessor::~CudaProcessor(void) {}

bool CudaProcessor::Initialize(void)
{
	m_hLUT = new uint8_t[MAX_PIXEL_VAL];
	
	if (!CudaCheck(cudaMalloc(&m_dLUT, MAX_PIXEL_VAL * sizeof(uint8_t))))
	{
		goto LB_FAILED_ALLOC_LUT;
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
	cudaFree(m_dLUT);
	delete[] m_hLUT;

LB_FAILED_ALLOC_LUT:
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
		
		avgBlur <<<grid, block >>> (m_dr2, m_dr, m_pitch, m_imgW, m_imgH);
		avgBlur <<<grid, block >>> (m_dg2, m_dg, m_pitch, m_imgW, m_imgH);
		avgBlur <<<grid, block >>> (m_db2, m_db, m_pitch, m_imgW, m_imgH);


		if (!CudaKernelCheck())
		{
			fprintf(stderr, "Apply blur failed\n");
			goto LB_FAILED_KERNEL;
		}
	}

	if (!CudaCheck(cudaMemcpy2D(ioBuf->r, m_imgW, m_dr2, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->g, (m_imgW), m_dg2, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}
	if (!CudaCheck(cudaMemcpy2D(ioBuf->b, (m_imgW), m_db2, m_pitch, static_cast<size_t>(m_imgW), static_cast<size_t>(m_imgH), cudaMemcpyDeviceToHost)))
	{
		goto LB_FAILED_MEMCPY_DTOH;
	}


	return true;

LB_FAILED_MEMCPY_DTOH:

LB_FAILED_KERNEL:

LB_FAILED_MEMCPY_HTOD:

	return false;
}

bool CudaProcessor::ApplyFuzzyContrast(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto)
{
	resetDevBufs();

	m_imgW = ioBuf->imgW;
	m_imgH = ioBuf->imgH;

	uint32_t* fuzzyH = fuzzyHistogram(inHisto, 8);
	uint32_t pl = calPartitioningLevel(fuzzyH, 0, MAX_PIXEL_VAL);

	memset(m_hLUT, 0, MAX_PIXEL_VAL * sizeof(uint8_t));

	uint64_t sumL = 0, sumU = 0;

	for (uint32_t i = 0; i < MAX_PIXEL_VAL; ++i)
	{
		if (i <= pl) { sumL += fuzzyH[i]; }
		else { sumU += fuzzyH[i]; }
	}

	double cumulateVal = 0.0;
	for (uint32_t i = 0; i <= pl; ++i)
	{
		cumulateVal += (double)fuzzyH[i] / sumL;
		m_hLUT[i] = (uint8_t)(pl * cumulateVal);
	}

	cumulateVal = 0.0;
	for (int i = pl + 1; i < MAX_PIXEL_VAL; ++i)
	{
		cumulateVal += (double)fuzzyH[i] / sumU;
		if (cumulateVal >= 1.0) cumulateVal = 1.0;

		m_hLUT[i] = (pl + 1) + (uint8_t)((MAX_PIXEL_VAL - pl - 2) * cumulateVal);
	}

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
		cudaMemset(m_dLUT, 0, MAX_PIXEL_VAL * sizeof(uint8_t));
		cudaMemcpy(m_dLUT, m_hLUT, MAX_PIXEL_VAL * sizeof(uint8_t), cudaMemcpyHostToDevice);
		delete[] fuzzyH;

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithLUT <<<grid, block>>> (m_dLUT, m_dr, m_pitch, m_imgW, m_imgH);
		replaceWithLUT <<<grid, block>>> (m_dLUT, m_dg, m_pitch, m_imgW, m_imgH);
		replaceWithLUT <<<grid, block>>> (m_dLUT, m_db, m_pitch, m_imgW, m_imgH);

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

LB_FAILED_MEMCPY_HTOD:
LB_FAILED_KERNEL:
LB_FAILED_MEMCPY_DTOH:

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
		const uint32_t DENOMINATOR = m_imgW * m_imgH - 1;

		float cdf = 0.0f; // cumulate count
		memset(m_hLUT, 0, MAX_PIXEL_VAL);

		for (uint32_t i = 0; i < MAX_PIXEL_VAL; ++i)
		{
			cdf += ioHisto[i];
			float h = (cdf / DENOMINATOR) * 255.0f;

			m_hLUT[i] = (uint8_t)(h + 0.5f);
		}

		cudaMemcpy(m_dLUT, m_hLUT, MAX_PIXEL_VAL, cudaMemcpyHostToDevice);

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithLUT << <grid, block >> > (m_dLUT, m_dr, m_pitch, m_imgW, m_imgH);
		replaceWithLUT << <grid, block >> > (m_dLUT, m_dg, m_pitch, m_imgW, m_imgH);
		replaceWithLUT << <grid, block >> > (m_dLUT, m_db, m_pitch, m_imgW, m_imgH);

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

	uint8_t* d_rgbLUT = nullptr;
	if (!CudaCheck(cudaMalloc(&d_rgbLUT, sizeof(uint8_t) * MAX_PIXEL_VAL * 3)));

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
		Cluster_s* cluster = isoDataClustering(inHisto);

		uint8_t rgbLUT[MAX_PIXEL_VAL * 3];

		while (cluster != nullptr)
		{
			uint32_t startIDX = cluster->startIDX;
			uint32_t endIDX = startIDX + cluster->length;

			uint8_t r = rand() % MAX_PIXEL_VAL;
			uint8_t g = rand() % MAX_PIXEL_VAL;
			uint8_t b = rand() % MAX_PIXEL_VAL;

			for (uint32_t i = startIDX; i < endIDX; ++i)
			{
				rgbLUT[i] = r;
				rgbLUT[i + (MAX_PIXEL_VAL * 1)] = g;
				rgbLUT[i + (MAX_PIXEL_VAL * 2)] = b;
			}

			cluster = cluster->next;
		}

		while (cluster != nullptr)
		{
			Cluster_s* next = cluster->next;
			delete cluster;
			cluster = next;
		}

		cudaMemcpy(d_rgbLUT, rgbLUT, sizeof(uint8_t) * MAX_PIXEL_VAL * 3, cudaMemcpyHostToDevice);

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithRGBLUT <<<grid, block >>> (d_rgbLUT, m_dr, m_dg, m_db, m_pitch, m_imgW, m_imgH);
		
		if (!CudaKernelCheck())
		{
			fprintf(stderr, "ColorMap kernel failed\n");
			goto LB_FAILED_KERNEL;
		}

		cudaFree(d_rgbLUT);
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

	cudaFree(d_rgbLUT);

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

		laplacian << <grid, block >> > (m_dr2, m_dr, m_pitch, m_imgW, m_imgH);
		laplacian << <grid, block >> > (m_dg2, m_dg, m_pitch, m_imgW, m_imgH);
		laplacian << <grid, block >> > (m_db2, m_db, m_pitch, m_imgW, m_imgH);

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

bool CudaProcessor::ApplyMultiFuzzyContrast(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto)
{
	resetDevBufs();

	const uint32_t IMAGE_PIXELS = ioBuf->imgW * ioBuf->imgH;

	uint32_t* fuzzyH = fuzzyHistogram(inHisto, 6);

	Cluster_s* cluster = isoDataClustering(fuzzyH);
	uint32_t clusterCount = 0;

	std::vector<Cluster_s*> reversCL;

	while (cluster != nullptr)
	{
		++clusterCount;
		reversCL.push_back(cluster);
		cluster = cluster->next;
	}
	
	cudaMemset(m_dLUT, 0, MAX_PIXEL_VAL * sizeof(uint8_t));
	memset(m_hLUT, 0, MAX_PIXEL_VAL * sizeof(uint8_t));


	uint8_t lastOutputValue = 0;

	for (uint32_t i = 0; i < clusterCount; ++i)
	{
		Cluster_s* currCL = reversCL[i];
		uint32_t endIDX = currCL->startIDX + currCL->length;
		uint32_t pl = calPartitioningLevel(fuzzyH, currCL->startIDX, currCL->length);

		float sumL = 0.0f, sumU = 0.0f;
		for (uint32_t j = currCL->startIDX; j < endIDX; ++j)
		{
			if (j <= pl) sumL += (float)fuzzyH[j];
			else sumU += (float)fuzzyH[j];
		}

		float cumulatedVal = 0.0f;
		for (uint32_t j = currCL->startIDX; j <= pl; ++j)
		{
			if (sumL > 0) cumulatedVal += (float)fuzzyH[j] / sumL;
			uint32_t rangeL = (pl > lastOutputValue) ? (pl - lastOutputValue) : 0;
			m_hLUT[j] = (uint8_t)(lastOutputValue + (rangeL * cumulatedVal) + 0.5f);
		}

		cumulatedVal = 0.0f;
		uint8_t plOutput = m_hLUT[pl]; 
		for (uint32_t j = pl + 1; j < endIDX; ++j)
		{
			if (sumU > 0) cumulatedVal += (float)fuzzyH[j] / sumU;

			uint32_t ts = plOutput + 1;
			uint32_t targetEnd = (i == clusterCount - 1) ? 255 : (endIDX - 1);
			uint32_t rangeU = (targetEnd > ts) ? (targetEnd - ts) : 0;

			m_hLUT[j] = (uint8_t)(ts + (rangeU * cumulatedVal) + 0.5f);
		}

		lastOutputValue = m_hLUT[endIDX - 1];
	}

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
		cudaMemcpy(m_dLUT, m_hLUT, MAX_PIXEL_VAL, cudaMemcpyHostToDevice);

		dim3 block(BLOCK, BLOCK);
		dim3 grid((m_imgW + BLOCK - 1) / BLOCK, (m_imgH + BLOCK - 1) / BLOCK);

		replaceWithLUT <<<grid, block>>> (m_dLUT, m_dr, m_pitch, m_imgW, m_imgH);
		replaceWithLUT <<<grid, block>>> (m_dLUT, m_dg, m_pitch, m_imgW, m_imgH);
		replaceWithLUT <<<grid, block>>> (m_dLUT, m_db, m_pitch, m_imgW, m_imgH);

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

LB_FAILED_MEMCPY_HTOD:
LB_FAILED_KERNEL:
LB_FAILED_MEMCPY_DTOH:

	return false;
}





void CudaProcessor::CloseCudaHandles(void)
{
	SAFE_RELEASE(m_dr)
	SAFE_RELEASE(m_dg)
	SAFE_RELEASE(m_db)

	SAFE_RELEASE(m_dr2)
	SAFE_RELEASE(m_dg2)
	SAFE_RELEASE(m_db2)
}










void CudaProcessor::resetDevBufs(void)
{
	static const int DBUF_SIZE = CU_IMG_H * CU_IMG_W;

	m_imgW = 0;
	m_imgH = 0;

	cudaMemset(m_dr, 0, DBUF_SIZE);
	cudaMemset(m_dg, 0, DBUF_SIZE);
	cudaMemset(m_db, 0, DBUF_SIZE);

	cudaMemset(m_dr2, 0, DBUF_SIZE);
	cudaMemset(m_dg2, 0, DBUF_SIZE);
	cudaMemset(m_db2, 0, DBUF_SIZE);


	if (!CudaKernelCheck())
	{
		fprintf(stderr, "cudaMemset failed \n");
#ifdef _DEBUG
		__debugbreak();
#endif
		exit(1);
	}
}

CudaProcessor::Cluster_s* CudaProcessor::isoDataClustering(uint32_t*& histo)
{
	Cluster_s* head = new Cluster_s;
	head->startIDX = 0;
	head->length = 0;
	head->next = nullptr;

	Cluster_s* current = head;

	for (uint32_t i = 0; i < MAX_PIXEL_VAL; ++i)
	{
		current->cv = calCV(histo, current->startIDX, current->length);

		if (current->cv > THRESHOLD && current->length > 1)
		{
			Cluster_s* newNode = new Cluster_s;
			newNode->startIDX = i;
			newNode->length = 0;
			newNode->next = nullptr;

			current->next = newNode;
			current = newNode;
		}

		current->length++;
	}

	return head;
}



uint32_t* CudaProcessor::fuzzyHistogram(uint32_t* inHisto, uint32_t theta)
{
	uint32_t* outHisto = new uint32_t[MAX_PIXEL_VAL];

	for (int i = 0; i < MAX_PIXEL_VAL; ++i)
	{
		float ratio = 0.0f;
		float sum = 0.0f;
		float weightSum = 0.0f;

		for (int j = i - (int)theta; j <= i + (int)theta; ++j)
		{
			if (j >= 0 && j < MAX_PIXEL_VAL)
			{
				ratio = (1 - (float)(std::fabs(j - i)) / theta);
				ratio = ratio < 0 ? 0 : ratio;
				
				sum += inHisto[j] * ratio;
				weightSum += ratio;
			}
		}

		outHisto[i] = (uint32_t)(sum / weightSum + 0.5f);
	}

	return outHisto;
}


uint32_t CudaProcessor::calPartitioningLevel(uint32_t*& fuzzyedHisto, uint32_t startIDX, uint32_t range)
{
	uint64_t totalPixels = 0;
	for (uint32_t i = 0; i < range; ++i)
		totalPixels += fuzzyedHisto[i];

	double numerator = 0.0, denominator = 0.0;
	for (uint32_t i = startIDX; i < startIDX + range; ++i)
	{
		double prob = (double)fuzzyedHisto[i] / (double)totalPixels;
		numerator += i * prob;
		denominator += prob;
	}

	float val = (float)(numerator / denominator);
	return (uint32_t)(val + 0.5f);
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










// CUDA codes

__device__ inline uint8_t getAvgWithUCHAR3(const uchar3& one, const uchar3& two, const uchar3& three)
{
	uint3 sum = make_uint3(one.x + two.x + three.x, one.y + two.y + three.y, one.z + two.z + three.z);
	uint32_t avg = (sum.x + sum.y + sum.z) / 9;
	return avg;
}

__global__ void avgBlur(uint8_t* dstBuf, uint8_t* srcBuf, size_t pitch, uint32_t width, uint32_t height)
{
	const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height ) return;

	int nsrcY = min(max(srcY - 1, 0), height - 1);
	int bsrcY = min(max(srcY + 1, 0), height - 1);
	int nsrcX = min(max(srcX - 1, 0), width - 1);

	uchar3 top = *(uchar3*)(srcBuf + nsrcX + pitch * nsrcY);
	uchar3 middle = *(uchar3*)(srcBuf + nsrcX + pitch * srcY);
	uchar3 bottom = *(uchar3*)(srcBuf + nsrcX + pitch * bsrcY);

	uint8_t val = getAvgWithUCHAR3(top, middle, bottom);

	dstBuf[srcX + pitch * srcY] = val;
}

__global__ void laplacian(uint8_t* dstBuf, uint8_t* srcBuf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width - 3 || srcY >= height) return;

	int nsrcY = min(max(srcY - 1, 0), height - 1);
	int bsrcY = min(max(srcY + 1, 0), height - 1);
	int nsrcX = min(max(srcX - 1, 0), width - 1);


	uchar3 top = *(uchar3*)(srcBuf + nsrcX + pitch * nsrcY);
	uchar3 middle = *(uchar3*)(srcBuf + nsrcX + pitch * srcY);
	uchar3 bottom = *(uchar3*)(srcBuf + nsrcX + pitch * bsrcY);

	int sum = top.y + middle.x + middle.y * -4 + middle.z + bottom.y;
	dstBuf[srcX + pitch * srcY] = min(max(sum, 0), 255);
}

__global__ void replaceWithLUT(uint8_t* LUT, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height) return;

	uint8_t pixelVal = buf[srcX + pitch * srcY];
	uint8_t equalizedVal = LUT[pixelVal];

	buf[srcX + pitch * srcY] = equalizedVal;
}

__global__ void replaceWithRGBLUT(uint8_t* rgbLUT, uint8_t* r, uint8_t* g, uint8_t* b, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height) return;

	uint8_t rVal, gVal, bVal;
	rVal = r[srcX + pitch * srcY];
	gVal = g[srcX + pitch * srcY];
	bVal = b[srcX + pitch * srcY];

	r[srcX + pitch * srcY] = rgbLUT[rVal];
	g[srcX + pitch * srcY] = rgbLUT[gVal + (256)];
	b[srcX + pitch * srcY] = rgbLUT[bVal + (256 * 2)];
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