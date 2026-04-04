#pragma once

#pragma pack(push, 1)

struct BmpFileHeader_s
{
	uint16_t fileMarker;
	uint32_t bufferSize;
	uint16_t unuse1;
	uint16_t unuse2;
	uint32_t imageDataOffet;
};

struct BmpInfoHeader_s
{
    uint32_t biSize;
    int      biWidth;
    int      biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int      biXPelsPerMeter;
    int      biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};

#pragma pack(pop)


namespace Bmp
{
	bool LoadRGBs(RAWImageBuf_s& imgB);
	bool StoreRGBs(const char* storeName, const RAWImageBuf_s& imgB);
}