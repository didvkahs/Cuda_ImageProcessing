#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <DirectXTex.h>
#include <EASTL/vector.h>

constexpr float DELAY_TIME = 0.05F;

struct Buf_s;

class D3DResources
{
public:

	D3DResources(void);
	~D3DResources(void);

	bool Initialize(HWND hWnd);
	void CloseD3DHandles(void);


	ID3D11Device*			GetDevice(void);
	ID3D11DeviceContext*	GetContext(void);
	ID3D11RenderTargetView* GetRTView(void);
	ID3D11VertexShader*		GetVShader(const int id);
	ID3D11PixelShader*		GetPShader(const int id);
	IDXGISwapChain*			GetSwapChain(void);
	ID3D11SamplerState*		GetSampler(void);
	ID3D11DepthStencilView* GetDepthStencilView(void);
	
	void AddVertexShader(LPCWSTR fileName, LPCSTR entryPoint, int& outID, ID3DBlob*& outBlob);
	void AddPixelShader(LPCWSTR fileName, LPCSTR entryPoint, int& outID);

	ID3DBlob*			BuildShaderBlob(const LPCWSTR fileName, const LPCSTR shaderModel);
	ID3D11InputLayout*	CreateInputLayout(const D3D11_INPUT_ELEMENT_DESC* layout, UINT numElement, ID3DBlob* vsBlob) const;


private:

	HRESULT compileShader(LPCWSTR fileName, LPCSTR entryPoint, LPCSTR shaderModel, ID3DBlob** blob);
	

	bool getMaxVideoAdapter(void);
	bool createDeviceAndSwapChain(void);
	bool createRenderTargets(void);
	void createClientSizeViewPort(void);


private:

	struct Texture_s
	{
		WCHAR* fileName = nullptr;
		DirectX::ScratchImage imageData;
		DirectX::TexMetadata  imageMetaData;
	};

	struct Vertex_s
	{
		DirectX::XMFLOAT2 pos;
		DirectX::XMFLOAT2 uv;
	};

private:

	HWND m_hWnd = {};

	UINT winW = 0, winH = 0;

	D3D_FEATURE_LEVEL m_featureLevel = D3D_FEATURE_LEVEL_11_0;

	IDXGISwapChain*			m_SwapChain = nullptr;
	ID3D11Device*			m_Device = nullptr;
	ID3D11DeviceContext*	m_DevContext = nullptr;
	ID3D11RenderTargetView* m_RTView = nullptr;
	IDXGIAdapter1*			m_Adapter = nullptr;

	eastl::vector<ID3D11VertexShader*> m_VertexShaders = {};
	eastl::vector<ID3D11PixelShader*>  m_PixelShaders = {};

	ID3D11SamplerState*				m_SamplerLinear = nullptr;
	ID3D11Texture2D*				m_DepthStencil = nullptr;
	ID3D11DepthStencilView*			m_DepthStencilView = nullptr;	
};