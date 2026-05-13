#include <cstdint>
#include "D3DResources.h"

#include "BmpBuffer.h"
#include "BmpReader.h"

#include "CudaProcessor.h"
#include "ImageProcessor.h"
#include <iostream>

#define SAFE_RELEASE(p) {if(p != nullptr) {p->Release(); p = nullptr;}}

constexpr int MAX_PIXEL_VALUE = 256;
constexpr int ImageProcessor_SIZE = MAX_PIXEL_VALUE * sizeof(uint32_t);


ImageProcessor::ImageProcessor(void) {};
ImageProcessor::~ImageProcessor(void) {};


bool ImageProcessor::Initialize(D3DResources& resource)
{
	m_ImgB = new RAWImageBuf_s;
	m_tempB = new RAWImageBuf_s;

	if (!Bmp::LoadRGBs(*m_ImgB))
	{
		fprintf(stderr, "LoadRGBs failed with error \n");
		goto LB_FAILED_LOAD_RGB;
	}

	{
		const uint32_t BUFSIZE = m_ImgB->imgH * m_ImgB->imgW;
		m_tempB->imgH = m_ImgB->imgH;
		m_tempB->imgW = m_ImgB->imgW;
		m_tempB->r = new uint8_t[BUFSIZE];
		m_tempB->g = new uint8_t[BUFSIZE];
		m_tempB->b = new uint8_t[BUFSIZE];
	}


	m_histoArr= new uint32_t[MAX_PIXEL_VALUE];
	memset(m_histoArr, 0, ImageProcessor_SIZE);

	m_cuprocess = new CudaProcessor();
	if (!m_cuprocess->Initialize())
	{
		fprintf(stderr, "cudaProcessor failed with error \n");
		goto LB_FAILED_CUDA_INITIALIZE;
	}
	
	
	m_Resource = resource;

	if (!createHistogramSRV())
	{
		fprintf(stderr, "createHistogramSRV failed\n");
		goto LB_FAILED_CREATE_HISTOGRAM;
	}

	if (!createEqualizedSRV())
	{
		fprintf(stderr, "createEqualizedSRV failed\n");
		goto LB_FAILED_CREATE_EQUALIZED;
	}

	if (!createFuzzyContrastSRV())
	{
		fprintf(stderr, "createFuzzyContrastSRV failed\n");
		goto LB_FAILED_FUZZY_CONTRAST;
	}

	if (!createAVGBlurSRV())
	{
		fprintf(stderr, "createAVGBlurSRV failed\n");
		goto LB_FAILED_CREATE_AVGBLUR;
	}

	if (!createColorMapSRV())
	{
		fprintf(stderr, "craeteColorMapSRV failed\n");
		goto LB_FAILED_CREATE_COLORMAP;
	}

	if (!createMultiFuzzyContrastSRV())
	{
		fprintf(stderr, "createMultiFuzzyContrast failed\n");
		goto LB_FAILED_CREATE_MULTI_FUZZY_CONTRAST;
	}

	if (!createLaplacianSRV())
	{
		fprintf(stderr, "createLaplacianSRV failed\n");
		goto LB_FAILED_CREATE_LAPLACIAN;
	}

	{
		D3D11_INPUT_ELEMENT_DESC layout[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
		};

		m_InputLayout = m_Resource.CreateInputLayout(layout, ARRAYSIZE(layout), m_vsBlob);

		createD3DBuffer();
	}

	return true;

	LB_FAILED_CREATE_LAPLACIAN:
	SAFE_RELEASE(m_SRVs[VS_MULTI_FUZZY])

	LB_FAILED_CREATE_MULTI_FUZZY_CONTRAST:
	SAFE_RELEASE(m_SRVs[VS_COLOR_MAP])

	LB_FAILED_CREATE_COLORMAP:
	SAFE_RELEASE(m_SRVs[VS_BLUR])

	LB_FAILED_CREATE_AVGBLUR:
	SAFE_RELEASE(m_SRVs[VS_FUZZY_CONTRAST])

	LB_FAILED_FUZZY_CONTRAST :
	SAFE_RELEASE(m_SRVs[VS_EQUALIZE])

	LB_FAILED_CREATE_EQUALIZED :
	SAFE_RELEASE(m_SRVs[VS_HISTOGRAM])

	LB_FAILED_CREATE_HISTOGRAM:
	m_cuprocess->CloseCudaHandles();
	m_cuprocess = nullptr;

    LB_FAILED_CUDA_INITIALIZE:
	delete[] m_tempB->r;
	m_tempB->r = nullptr;
	delete[] m_tempB->g;
	m_tempB->g = nullptr;
	delete[] m_tempB->b;
	m_tempB->b = nullptr;
	delete m_tempB;
	m_tempB = nullptr;


	delete[] m_rgbHisto;
	delete[] m_histoArr;
	m_histoArr = nullptr;
	free(m_ImgB->r);
	m_ImgB = nullptr;
	free(m_ImgB->g);
	m_ImgB->g = nullptr;
	free(m_ImgB->b);
	m_ImgB->b = nullptr;
	delete m_ImgB;
	m_ImgB = nullptr;

    LB_FAILED_LOAD_RGB:
	return false;
}

void ImageProcessor::SetD3DResource(D3DResources& resource)
{
	m_Resource = resource;
}

void ImageProcessor::SetVSID(const VertexShaderID_e id)
{
	m_currVSID = id;
}

void ImageProcessor::SetPSID(const PixelShaderID_e id)
{
	m_currPSID = id;
}

void ImageProcessor::SetupD3D(void)
{
	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
	};

	m_InputLayout = m_Resource.CreateInputLayout(layout, ARRAYSIZE(layout), m_vsBlob);

	createD3DBuffer();
}

void ImageProcessor::Draw(void)
{
	float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	
	ID3D11DeviceContext* devcon = m_Resource.GetContext();
	ID3D11RenderTargetView* rtview = m_Resource.GetRTView();
	ID3D11DepthStencilView* depthStencilView = m_Resource.GetDepthStencilView();
	ID3D11SamplerState* sampler = m_Resource.GetSampler();

	devcon->ClearRenderTargetView(rtview, clearColor);
	devcon->ClearDepthStencilView(depthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
	
	UINT stride = sizeof(HistoVertex_s);
	UINT offset = 0;
	
	devcon->IASetVertexBuffers(0, 1, &m_VertexBuffer, &stride, &offset);
	devcon->IASetInputLayout(m_InputLayout);
	devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	
	ID3D11VertexShader* targetVShader = m_Resource.GetVShader(m_vsIDs[m_currVSID]);
	ID3D11PixelShader* targetPShader = m_Resource.GetPShader(m_psIDs[m_currPSID]);
	
	devcon->VSSetShader(targetVShader, nullptr, 0);
	devcon->PSSetShader(targetPShader, nullptr, 0);
	devcon->PSSetShaderResources(0, 1, &m_SRVs[m_currVSID]);
	devcon->PSSetSamplers(0, 1, &sampler);
	
	devcon->Draw(6, 0);
}

void ImageProcessor::CloseImgProcessHandles(void)
{
	if (m_VertexBuffer) { m_VertexBuffer->Release(); }
	if (m_InputLayout) { m_InputLayout->Release(); }
	if (m_vsBlob) { m_vsBlob->Release(); }


	for (int i = 0; i < VS_SHADER_COUNT; ++i)
	{
		if (m_SRVs[i] != nullptr)
		{
			m_SRVs[i]->Release();
		}
	}

	if (m_ImgB->imgH)
	{
		free(m_ImgB->r);
		free(m_ImgB->g);
		free(m_ImgB->b);
	}
	delete m_ImgB;
	delete[] m_histoArr;

	m_cuprocess->CloseCudaHandles();
}






bool ImageProcessor::createHistogramSRV(void)
{
	computeHistogram();

	createSRV(nullptr, m_histoArr, VS_HISTOGRAM);
	m_Resource.AddVertexShader(L"HistogramShader.fx", "vsMain", m_vsIDs[VS_HISTOGRAM], m_vsBlob);
	m_Resource.AddPixelShader(L"HistogramShader.fx", "psMain", m_psIDs[PS_HISTOGRAM]);

	return true;
}

bool ImageProcessor::createEqualizedSRV(void)
{
	ID3DBlob* vsBlob = nullptr;

	const uint32_t BUFSIZE = m_ImgB->imgW * m_ImgB->imgH;
	memcpy(m_tempB->r, m_ImgB->r, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->g, m_ImgB->g, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->b, m_ImgB->b, BUFSIZE * sizeof(uint8_t));

	if (!m_cuprocess->ApplyEqualize(m_tempB, m_histoArr))
	{
		return false;
	}
	createSRV(m_tempB, nullptr, VS_EQUALIZE);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_EQUALIZE], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_EQUALIZE]);
	
	vsBlob->Release();
	return true;
}

bool ImageProcessor::createFuzzyContrastSRV(void)
{
	ID3DBlob* vsBlob = nullptr;
	
	if (!m_cuprocess->ApplyFuzzyContrast(m_ImgB, m_histoArr))
	{
		return false;
	}
	createSRV(m_ImgB, nullptr, VS_FUZZY_CONTRAST);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_FUZZY_CONTRAST], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_FUZZY_CONTRAST]);

	vsBlob->Release();
	return true;
}

bool ImageProcessor::createAVGBlurSRV(void)
{
	ID3DBlob* vsBlob = nullptr;
	const uint32_t BUFSIZE = m_ImgB->imgW * m_ImgB->imgH;

	memcpy(m_tempB->r, m_ImgB->r, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->g, m_ImgB->g, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->b, m_ImgB->b, BUFSIZE * sizeof(uint8_t));

	if (!m_cuprocess->ApplyBlur(m_tempB))
	{
		return false;
	}
	createSRV(m_tempB, nullptr, VS_BLUR);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_BLUR], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_BLUR]);

	vsBlob->Release();
	return true;
}

bool ImageProcessor::createColorMapSRV(void)
{
	ID3DBlob* vsBlob = nullptr;
	const uint32_t BUFSIZE = m_ImgB->imgW * m_ImgB->imgH;

	memcpy(m_tempB->r, m_ImgB->r, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->g, m_ImgB->g, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->b, m_ImgB->b, BUFSIZE * sizeof(uint8_t));

	if (!m_cuprocess->ApplyColorMap(m_tempB, m_histoArr))
	{
		return false;
	}
	createSRV(m_tempB, nullptr, VS_COLOR_MAP);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_COLOR_MAP], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_COLOR_MAP]);

	return true;
}

bool ImageProcessor::createMultiFuzzyContrastSRV(void)
{
	ID3DBlob* vsBlob = nullptr;
	const uint32_t BUFSIZE = m_ImgB->imgW * m_ImgB->imgH;

	memcpy(m_tempB->r, m_ImgB->r, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->g, m_ImgB->g, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->b, m_ImgB->b, BUFSIZE * sizeof(uint8_t));

	if (!m_cuprocess->ApplyMultiFuzzyContrast(m_tempB, m_histoArr))
	{
		return false;
	}
	createSRV(m_tempB, nullptr, VS_MULTI_FUZZY);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_MULTI_FUZZY], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_MULTI_FUZZY]);

	vsBlob->Release();

	return true;
}

bool ImageProcessor::createLaplacianSRV(void)
{
	ID3DBlob* vsBlob = nullptr;
	const uint32_t BUFSIZE = m_ImgB->imgW * m_ImgB->imgH;

	memcpy(m_tempB->r, m_ImgB->r, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->g, m_ImgB->g, BUFSIZE * sizeof(uint8_t));
	memcpy(m_tempB->b, m_ImgB->b, BUFSIZE * sizeof(uint8_t));

	if (!m_cuprocess->ApplyLaplacian(m_tempB))
	{
		return false;
	}
	createSRV(m_tempB, nullptr, VS_LAPLACIAN);
	m_Resource.AddVertexShader(L"ImageShader.fx", "vsMain", m_vsIDs[VS_LAPLACIAN], vsBlob);
	m_Resource.AddPixelShader(L"ImageShader.fx", "psMain", m_psIDs[PS_LAPLACIAN]);

	return true;
}









void ImageProcessor::computeHistogram(void)
{
	const uint32_t TOTAL_PIXEL = m_ImgB->imgW * m_ImgB->imgH;

	m_rgbHisto = new uint32_t[MAX_PIXEL_VALUE * 3];
	memset(m_rgbHisto, 0, sizeof(uint32_t) * MAX_PIXEL_VALUE * 3);
	uint32_t rVal, gVal, bVal;

	for (uint32_t i = 0; i < TOTAL_PIXEL; ++i)
	{
		rVal = m_ImgB->r[i];
		gVal = m_ImgB->g[i];
		bVal = m_ImgB->b[i];

		++m_rgbHisto[rVal + (MAX_PIXEL_VALUE * 0)];
		++m_rgbHisto[gVal + (MAX_PIXEL_VALUE * 1)];
		++m_rgbHisto[bVal + (MAX_PIXEL_VALUE * 2)];
	}


	float sum = 0;
	for (uint32_t i = 0; i < MAX_PIXEL_VALUE; ++i)
	{
		// NOTICE : use REC.601

		sum = (float)(m_rgbHisto[i]) * 0.299f + (float)(m_rgbHisto[i + (MAX_PIXEL_VALUE)]) * 0.587f + (float)(m_rgbHisto[i + MAX_PIXEL_VALUE * 2]) * 0.114f;
		m_histoArr[i] = (uint32_t)(sum + 0.5f);
	}
}




void ImageProcessor::createSRV(const RAWImageBuf_s* img, uint32_t* histoArr, VertexShaderID_e vsID)
{
	ID3D11Device* device = m_Resource.GetDevice();
	ID3D11Texture2D* texture = nullptr;
	D3D11_TEXTURE2D_DESC texd = {};
	D3D11_SUBRESOURCE_DATA initData = {};

	uint8_t* imgBuf = nullptr;

	if (img != nullptr)
	{
		uint32_t imgW = img->imgW;
		uint32_t imgH = img->imgH;

		imgBuf = new uint8_t[imgW * 4 * imgH];

		for (uint32_t i = 0; i < imgW * imgH; ++i)
		{
			imgBuf[i * 4 + 0] = img->r[i];
			imgBuf[i * 4 + 1] = img->g[i];
			imgBuf[i * 4 + 2] = img->b[i];
			imgBuf[i * 4 + 3] = 255;
		}

		texd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texd.Width = imgW;
		texd.Height = imgH;
		
		initData.pSysMem = imgBuf;
		initData.SysMemPitch = imgW * 4;
	}
	else if(histoArr != nullptr)
	{
		uint32_t imgW = MAX_PIXEL_VALUE;
		uint32_t imgH = MAX_PIXEL_VALUE;

		imgBuf = new uint8_t[imgW * imgH];

		uint32_t mode = 0;

		for (uint32_t i = 0; i < MAX_PIXEL_VALUE; ++i)
		{
			if (mode < m_histoArr[i])
			{
				mode = m_histoArr[i];
			}
		}

		for (uint32_t i = 0; i < MAX_PIXEL_VALUE; ++i)
		{
			float ratio = static_cast<float>(m_histoArr[i]) / static_cast<float>(mode);
			uint32_t barHeight = static_cast<uint32_t>(ratio * (MAX_PIXEL_VALUE - 1));

			for (uint32_t j = 0; j < MAX_PIXEL_VALUE; ++j)
			{
				if (j >= (MAX_PIXEL_VALUE - 1 - barHeight)) {
					imgBuf[i + j * MAX_PIXEL_VALUE] = 0;
				}
				else {
					imgBuf[i + j * MAX_PIXEL_VALUE] = 255;
				}
			}
		}

		texd.Format = DXGI_FORMAT_R8_UNORM;
		texd.Width = MAX_PIXEL_VALUE;
		texd.Height = MAX_PIXEL_VALUE;

		initData.pSysMem = imgBuf;
		initData.SysMemPitch = MAX_PIXEL_VALUE;
	}

	texd.MipLevels = 1;
	texd.ArraySize = 1;
	texd.SampleDesc.Count = 1;
	texd.Usage = D3D11_USAGE_DEFAULT;
	texd.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	HRESULT hr = device->CreateTexture2D(&texd, &initData, &texture);

	if (FAILED(hr))
	{
		fprintf(stderr, "create texture 2d failed \n");
#ifdef _DEBUG
		__debugbreak();
#endif
	}

	D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
	srvd.Format = texd.Format;
	srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvd.Texture2D.MipLevels = 1;
	srvd.Texture2D.MostDetailedMip = 0;

	hr = device->CreateShaderResourceView(texture, &srvd, &m_SRVs[vsID]);
	if (FAILED(hr))
	{
		fprintf(stderr, "create shader resource view failed\n");
#ifdef _DEBUG
		__debugbreak();
#endif
	}

	texture->Release();
	delete[] imgBuf;
	return;
}

void ImageProcessor::createD3DBuffer(void)
{
	HRESULT result = S_OK;

	using namespace DirectX;
	
	ID3D11Device* device = m_Resource.GetDevice();

	HistoVertex_s vertices[] = {
		{ XMFLOAT2(-1.0f,  1.0f), XMFLOAT2(0.0f, 0.0f) },
		{ XMFLOAT2(1.0f,  1.0f), XMFLOAT2(1.0f, 0.0f) },
		{ XMFLOAT2(-1.0f, -1.0f), XMFLOAT2(0.0f, 1.0f) },
		{ XMFLOAT2(-1.0f, -1.0f), XMFLOAT2(0.0f, 1.0f) },
		{ XMFLOAT2(1.0f,  1.0f), XMFLOAT2(1.0f, 0.0f) },
		{ XMFLOAT2(1.0f, -1.0f), XMFLOAT2(1.0f, 1.0f) }
	};

	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(HistoVertex_s) * 6;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

	D3D11_SUBRESOURCE_DATA initData = {};
	initData.pSysMem = vertices;

	result = device->CreateBuffer(&bd, &initData, &m_VertexBuffer);
	if (FAILED(result))
	{
#ifdef _DEBUG
		__debugbreak();
#endif
	}
}
