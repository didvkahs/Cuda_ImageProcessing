#pragma once

struct RAWImageBuf_s;

class D3DResources;
class CudaProcessor;


enum PixelShaderID_e
{
	PS_HISTOGRAM,
	PS_BLUR,
	PS_EQUALIZE,
	PS_COLOR_MAP,
	PS_LAPLACIAN,
	PS_SHADER_COUNT,
	PS_NON
};

enum VertexShaderID_e
{
	VS_HISTOGRAM,
	VS_BLUR,
	VS_EQUALIZE,
	VS_COLOR_MAP,
	VS_LAPLACIAN,
	VS_SHADER_COUNT,
	VS_NON
};


class ImageProcessor
{
public:

	ImageProcessor(void);
	~ImageProcessor(void);

	bool Initialize(D3DResources& resource);

	void SetD3DResource(D3DResources& resoruce);
	void SetVSID(const VertexShaderID_e id);
	void SetPSID(const PixelShaderID_e id);

	void SetupD3D(void);
	void Draw(void);
	
	void CloseImgProcessHandles(void);

private:
	struct HistoVertex_s
	{
		DirectX::XMFLOAT2 pos;
		DirectX::XMFLOAT2 tex;
	};

private:

	bool createHistogramSRV(void);
	bool createEqualizedSRV(void);
	bool createAVGBlurSRV(void);
	bool createColorMapSRV(void);
	bool createLaplacianSRV(void);


	void createSRV(const RAWImageBuf_s* img, uint32_t* histoArr, VertexShaderID_e vsID);
	void createD3DBuffer(void);

private:

	D3DResources m_Resource;

	ID3D11InputLayout* m_InputLayout = nullptr;

	ID3D11ShaderResourceView* m_SRVs[VS_SHADER_COUNT] = {};

	ID3D11Buffer* m_VertexBuffer = nullptr;
	ID3D11Buffer* m_PixelBuffer = nullptr;
	ID3DBlob* m_vsBlob = nullptr;

	int m_psIDs[PS_SHADER_COUNT];
	int m_vsIDs[VS_SHADER_COUNT];

	VertexShaderID_e m_currVSID = VS_NON;
	PixelShaderID_e m_currPSID = PS_NON;

	CudaProcessor* m_cuprocess = nullptr;

	uint32_t* m_histoArr = nullptr;
	RAWImageBuf_s* m_ImgB;
	RAWImageBuf_s* m_tempB;
};