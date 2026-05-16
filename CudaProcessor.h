#pragma once


class CudaProcessor
{
public:

	CudaProcessor(void);
	~CudaProcessor(void);

	bool Initialize(void);

	bool ApplyBlur(RAWImageBuf_s*& ioBuf);
	bool ApplyFuzzyContrast(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);
	bool ApplyEqualize(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);
	bool ApplyColorMap(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);
	bool ApplyLaplacian(RAWImageBuf_s*& ioBuf);
	bool ApplyMultiFuzzyContrast(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);
	bool ApplyDFT(RAWImageBuf_s*& inBuf, RAWImageBuf_s*& outBuf);

	void CloseCudaHandles(void);

private:
	struct Cluster_s
	{
		uint32_t startIDX = 0;
		uint32_t length = 0;

		float cv = 0.0f;

		Cluster_s* next = nullptr;
	};

private:
	void resetDevBufs(void);
	Cluster_s* isoDataClustering(uint32_t*& histo);

	void logRemap(uint8_t* dstBuf, float* magBuf);
	uint32_t* fuzzyHistogram(uint32_t* inHisto, uint32_t theta);
	uint32_t calPartitioningLevel(uint32_t*& fuzzyedHisto, uint32_t startIDX, uint32_t range);
	inline float calCV(uint32_t* histo, uint32_t startIDX, uint32_t length);

private:
	uint8_t* m_dLUT = nullptr;
	uint8_t* m_hLUT = nullptr;

	size_t m_pitch = 0;
	uint8_t* m_dr = nullptr;
	uint8_t* m_dg = nullptr;
	uint8_t* m_db = nullptr;

	uint8_t* m_dr2 = nullptr;
	uint8_t* m_dg2 = nullptr;
	uint8_t* m_db2 = nullptr;

	uint32_t m_imgW = 0;
	uint32_t m_imgH = 0;
};