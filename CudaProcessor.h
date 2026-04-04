#pragma once

class CudaProcessor
{
public:

	CudaProcessor(void);
	~CudaProcessor(void);

	bool Initialize(void);

	bool ComputeHistogram(RAWImageBuf_s*& inBuf, uint32_t*& outHisto);
	bool ApplyBlur(RAWImageBuf_s*& ioBuf);
	bool ApplyEqualize(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);
	bool ApplyColorMap(RAWImageBuf_s*& ioBuf, uint32_t*& inHisto);

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
	void isoDataClustering(uint32_t*& hiso);

	inline float calCV(uint32_t* histo, uint32_t startIDX, uint32_t length);

private:

	Cluster_s* m_clusters = nullptr;

	uint32_t* m_dHisto = nullptr;

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