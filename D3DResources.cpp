
#include <iostream>

#include <cstdint>
#include "BmpBuffer.h"
#include "BmpReader.h"
#include "D3DResources.h"

#define SAFE_RELEASE(p) {if(p != nullptr) {p->Release(); p = nullptr;}}

using namespace eastl;
using namespace DirectX;


void* __cdecl operator new[](unsigned __int64 size, char const*, int, unsigned int, char const*, int)
{
	return new unsigned char[size];
}
void* __cdecl operator new[](unsigned __int64 size, unsigned __int64, unsigned __int64, char const*, int, unsigned int, char const*, int)
{
	return new unsigned char[size];
}
namespace eastl {

	void __cdecl AssertionFailure(void*, const char* expression) {
#ifdef _DEBUG
		OutputDebugStringA(expression);
		__debugbreak();
#else
#endif
	}
}


D3DResources::D3DResources(void) {}
D3DResources::~D3DResources(void) {}


bool D3DResources::Initialize(HWND hWnd)
{
	m_hWnd = hWnd;

	RECT rect;
	GetClientRect(hWnd, &rect);

	winW = rect.right - rect.left;
	winH = rect.bottom - rect.top;
	
	

	if (!getMaxVideoAdapter())
	{
		fprintf(stderr, "getMaxVideoAdapter failed with error \n");
		goto LB_FAILED_CREATE_ADAPTER;
	}

	if (!createDeviceAndSwapChain())
	{
		fprintf(stderr, "createDeviceAndSwapChain failed with error\n");
		goto LB_FAILED_CREATE_DEVICE_AND_SWAPCHAIN;
	}
	
	if (!createRenderTargets())
	{
		fprintf(stderr, "createRenderTargets failed with error \n");
		goto LB_FAILED_CREAETE_RENDER_TARGETS;
	}
	
	{
		HRESULT hr = S_OK;
			
		D3D11_SAMPLER_DESC samd = {};
		samd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		samd.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
		samd.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
		samd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
		samd.ComparisonFunc = D3D11_COMPARISON_NEVER;
		samd.MinLOD = 0;
		samd.MaxLOD = D3D11_FLOAT32_MAX;

		hr = m_Device->CreateSamplerState(&samd, &m_SamplerLinear);

		if (FAILED(hr))
		{
			fprintf(stderr, "create sampler state failed\n");
			goto LB_FAILED_CREATE_SAMPLER;
		}
	}

	createClientSizeViewPort();

return true;

LB_FAILED_CREATE_BMP_TEXTURE:
	SAFE_RELEASE(m_SamplerLinear)

LB_FAILED_CREATE_SAMPLER:

	SAFE_RELEASE(m_RTView)
	SAFE_RELEASE(m_DepthStencil)
	SAFE_RELEASE(m_DepthStencilView)

LB_FAILED_CREAETE_RENDER_TARGETS:
	SAFE_RELEASE(m_Device)
	SAFE_RELEASE(m_DevContext)
	SAFE_RELEASE(m_SwapChain)

LB_FAILED_CREATE_DEVICE_AND_SWAPCHAIN:
	SAFE_RELEASE(m_Adapter)

LB_FAILED_CREATE_ADAPTER:
	return false;
}

void D3DResources::CloseD3DHandles()
{
	if (m_DevContext)
	{
		m_DevContext->ClearState();
		m_DevContext->Flush();
	}

	for (auto& vshader : m_VertexShaders)
	{
		if (vshader) { vshader->Release(); }
	}
	m_VertexShaders.clear();
	m_VertexShaders.shrink_to_fit();

	for (auto& pshader : m_PixelShaders)
	{
		if (pshader) { pshader->Release(); }
	}
	m_PixelShaders.clear();
	m_PixelShaders.shrink_to_fit();

	SAFE_RELEASE(m_DepthStencilView);
	SAFE_RELEASE(m_DepthStencil);
	SAFE_RELEASE(m_RTView);
	SAFE_RELEASE(m_SamplerLinear);
	SAFE_RELEASE(m_SwapChain);

#ifdef _DEBUG
	ID3D11Debug* debugDev = nullptr;
	if (m_Device && SUCCEEDED(m_Device->QueryInterface(__uuidof(ID3D11Debug), (void**)&debugDev)))
	{
		debugDev->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
		debugDev->Release();
	}
#endif

	SAFE_RELEASE(m_DevContext);
	SAFE_RELEASE(m_Device);
	SAFE_RELEASE(m_Adapter);
}


ID3D11Device* D3DResources::GetDevice(void)
{
	return m_Device;
}

ID3D11DeviceContext* D3DResources::GetContext(void)
{
	return m_DevContext;
}

ID3D11RenderTargetView* D3DResources::GetRTView(void)
{
	return m_RTView;
}

ID3D11VertexShader* D3DResources::GetVShader(const int id)
{
	return m_VertexShaders.at(id);
}

ID3D11PixelShader* D3DResources::GetPShader(const int id)
{
	return m_PixelShaders.at(id);
}

ID3D11SamplerState* D3DResources::GetSampler(void)
{
	return m_SamplerLinear;
}

IDXGISwapChain* D3DResources::GetSwapChain(void)
{
	return m_SwapChain;
}

ID3D11DepthStencilView* D3DResources::GetDepthStencilView(void)
{
	return m_DepthStencilView;
}

void D3DResources::AddVertexShader(LPCWSTR fileName, LPCSTR entryPoint, int& outID, ID3DBlob*& outBlob)
{
	ID3D11VertexShader* vshader = nullptr;

	HRESULT result = compileShader(fileName, entryPoint, "vs_5_0", &outBlob);
	if (FAILED(result))
	{
		fprintf(stderr, "VS compile shader failed with error : NO_SUCH_SHADER | INVALID_ENTRY\n");
#ifdef _DEBUG
		__debugbreak();
#endif
		exit(1);
	}

	result = m_Device->CreateVertexShader(outBlob->GetBufferPointer(), outBlob->GetBufferSize(), nullptr, &vshader);
	if (FAILED(result))
	{
		fprintf(stderr, "VS create shader failed \n");
#ifdef _DEBUG
			__debugbreak();
#endif
		exit(1);
	}

	m_VertexShaders.push_back(std::move(vshader));
	outID = m_VertexShaders.size() - 1;
}

void D3DResources::AddPixelShader(LPCWSTR fileName, LPCSTR entryPoint, int& outID)
{
	ID3DBlob* psBlob = nullptr;
	ID3D11PixelShader* pshader = nullptr;

	HRESULT result = compileShader(fileName, entryPoint, "ps_5_0", &psBlob);
	if (FAILED(result))
	{
		fprintf(stderr, "PS compileShader failed with error : NO_SUCH_FILE | INVALID_ENTERY\n");
#ifdef _DEBUG
		__debugbreak();
#endif
		exit(1);
	}

	result = m_Device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pshader);
	if (FAILED(result))
	{
		fprintf(stderr, "PS createShader failed\n");
		SAFE_RELEASE(psBlob)
#ifdef _DEBUG
		__debugbreak();
#endif
		exit(1);
	}

	m_PixelShaders.push_back(std::move(pshader));
	outID = m_PixelShaders.size() - 1;

	SAFE_RELEASE(psBlob)
}


ID3DBlob* D3DResources::BuildShaderBlob(const LPCWSTR fileName, LPCSTR shaderModel)
{
	HRESULT result = S_FALSE;
	ID3DBlob* shaderBlob = nullptr;

	const LPCSTR entryPoint = "main";
	result = compileShader(fileName, entryPoint, shaderModel, &shaderBlob);

	if (FAILED(result))
	{
		fprintf(stderr, "compileShader failed with error : INVALID_FILE_NAME | INVALID_SHADER_MODEL\n");
		SAFE_RELEASE(shaderBlob)
	}

	return shaderBlob;
}

ID3D11InputLayout* D3DResources::CreateInputLayout(const D3D11_INPUT_ELEMENT_DESC* layout,UINT numElement, ID3DBlob* vsBlob) const
{
	HRESULT result = S_FALSE;
	ID3D11InputLayout* outLayout = nullptr;

	result = m_Device->CreateInputLayout(layout, numElement, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &outLayout);
	if (FAILED(result))
	{
		fprintf(stderr, "createInputLayout failed with error : INVALID_LAYOUT | INVALID_NUM_ELEMENT | INVALID_BLOB\n");
		SAFE_RELEASE(outLayout)
	}

	return outLayout;
}












bool D3DResources::getMaxVideoAdapter(void)
{
	IDXGIFactory1* pFactory = nullptr;
	UINT adapterIdx = 0;

	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory)))
	{
		return false;
	}

	DXGI_ADAPTER_DESC1 adapterDesc;
	size_t maxVideoMemSize = 0;
	IDXGIAdapter1* adapter;
	for (UINT i = 0; pFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
	{
		adapter->GetDesc1(&adapterDesc);
		if (adapterDesc.DedicatedVideoMemory > maxVideoMemSize)
		{
			adapterIdx = i;
			maxVideoMemSize = adapterDesc.DedicatedVideoMemory;
		}
		adapter->Release();
	}

	HRESULT hr = pFactory->EnumAdapters1(adapterIdx, &m_Adapter);
	pFactory->Release();
	if (hr == DXGI_ERROR_NOT_FOUND)
	{
		return false;
	}
	return true;
}

bool D3DResources::createDeviceAndSwapChain(void)
{
	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.BufferCount = 1;
	sd.BufferDesc.Width = winW;
	sd.BufferDesc.Height = winH;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 0;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = m_hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;

	D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0
	};

	UINT createDeviceFlag = 0;
#if defined(_DEBUG)
	createDeviceFlag |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	HRESULT hr = D3D11CreateDeviceAndSwapChain(m_Adapter, D3D_DRIVER_TYPE_UNKNOWN,
		NULL, createDeviceFlag, featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION,
		&sd, &m_SwapChain, &m_Device, &m_featureLevel, &m_DevContext);

	if (FAILED(hr))
	{
		return false;
	}
	return true;
}

bool D3DResources::createRenderTargets(void)
{
	ID3D11Texture2D* pBackBuffer = nullptr;
	HRESULT hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pBackBuffer);
	if (FAILED(hr))
	{
		return false;
	}

	hr = m_Device->CreateRenderTargetView(pBackBuffer, nullptr, &m_RTView);
	if (FAILED(hr))
	{
		pBackBuffer->Release();
		goto LB_FAILED_CREATE_RENDER_TARGET_VIEW;
	}
	pBackBuffer->Release();


	D3D11_TEXTURE2D_DESC descDepth;
	ZeroMemory(&descDepth, sizeof(descDepth));
	descDepth.Width = winW;
	descDepth.Height = winH;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	descDepth.SampleDesc.Count = 1;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D11_USAGE_DEFAULT;
	descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	descDepth.CPUAccessFlags = 0;
	descDepth.MiscFlags = 0;
	hr = m_Device->CreateTexture2D(&descDepth, nullptr, &m_DepthStencil);
	if (FAILED(hr))
	{
		fprintf(stderr, "create depthStencil failed\n");
		goto LB_FAILED_CREATE_DEPTH_STENCIL;
	}

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory(&descDSV, sizeof(descDSV));
	descDSV.Format = descDepth.Format;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;

	hr = m_Device->CreateDepthStencilView(m_DepthStencil, &descDSV, &m_DepthStencilView);
	if (FAILED(hr))
	{
		goto LB_FAILED_CREATE_DEPTH_STENCIL_VIEW;
	}
	m_DevContext->OMSetRenderTargets(1, &m_RTView, m_DepthStencilView);

	return true;
LB_FAILED_CREATE_DEPTH_STENCIL_VIEW:
	m_DepthStencil->Release();
LB_FAILED_CREATE_DEPTH_STENCIL:
	m_RTView->Release();
LB_FAILED_CREATE_RENDER_TARGET_VIEW:
	return false;
}



HRESULT D3DResources::compileShader(LPCWSTR fileName, LPCSTR entryPoint, LPCSTR shaderModel, ID3DBlob** blob)
{
	HRESULT result = S_FALSE;
	ID3DBlob* errorBlob = nullptr;

	UINT shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
	shaderFlags |= D3DCOMPILE_DEBUG;
#endif

	result = D3DCompileFromFile(fileName, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, shaderModel, shaderFlags, 0, blob, &errorBlob);

	if (FAILED(result))
	{
		fprintf(stderr, "complieFromFile failed with error : %s\n", (char*)errorBlob->GetBufferPointer());

		if (*blob) { (*blob)->Release();}
		goto LB_RETURN;
	}

	if (errorBlob) { errorBlob->Release(); }
	result = S_OK;

LB_RETURN:

	return result;
}

void D3DResources::createClientSizeViewPort(void)
{
	D3D11_VIEWPORT vp = {};
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.Width = (FLOAT)winW;
	vp.Height = (FLOAT)winH;
	vp.MinDepth = 0;
	vp.MaxDepth = 1.0f;

	m_DevContext->RSSetViewports(1, &vp);
}