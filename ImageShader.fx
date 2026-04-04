struct VS_INPUT
{
    float4 pos : POSITION;
    float2 uv : TEXCOORD0;
};

struct PS_INPUT
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2DArray g_texture : register(t0);
SamplerState g_sampler : register(s0);

PS_INPUT vsMain(VS_INPUT input)
{
    PS_INPUT output;
    
    output.pos = input.pos;
    output.uv = input.uv;
    
    return output;
}

float4 psMain(PS_INPUT input) : SV_Target
{
    static const float MAX_PIXEL_VALUE = 1.0f;
    
    float4 color = g_texture.Sample(g_sampler, float3(input.uv, 0));
        
    return color;
}