#include <cstdint>

#include "BmpBuffer.h"
#include "BmpReader.h"

#include <iostream>
#include <cstdlib>
#include <fstream>



#define BMPROWSIZE(width) ((width * 3 + 3) & (~3)) // for 4 byte alligne, remove under 2 bits

constexpr unsigned short BMP_MARKER = 0x4D42;
constexpr int FILE_MARKER_SIZE = 14;
constexpr int FILE_BODY_SIZE = 40;
constexpr int BIT_PER_PIXEL = 24;


namespace Bmp
{
	bool LoadRGBs(RAWImageBuf_s& imgB)
	{
		bool result = false;

		std::ifstream file("./highWay1.bmp", std::ios::binary);

		if (!file)
		{
			fprintf(stderr, "file failed with error : INVALID_FILE_NAME | FILE_NOT_EXIST\n");
			return false;
		}

		BmpFileHeader_s fileHeader;
		file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
		if (fileHeader.fileMarker != BMP_MARKER)
		{
			fprintf(stderr, "read failed with error : INVALID_FILE_TYPE | NO_FILE_MARKER\n");
			file.close();
			return false;
		}

		BmpInfoHeader_s infoHeader;
		file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

		uint32_t width = infoHeader.biWidth;
		uint32_t height = infoHeader.biHeight;

		imgB.imgW = width;
		imgB.imgH = height;

		if (infoHeader.biBitCount != BIT_PER_PIXEL)
		{
			fprintf(stderr, "invalid bitPix, unsupported file format\n");
			file.close();
			return false;
		}


		const uint32_t bufferSize = width * height;


		imgB.r = (uint8_t*)malloc(sizeof(uint8_t) * (bufferSize));
		imgB.g = (uint8_t*)malloc(sizeof(uint8_t) * (bufferSize));
		imgB.b = (uint8_t*)malloc(sizeof(uint8_t) * (bufferSize));

		unsigned int rowSize = BMPROWSIZE(width); 
		char* tempBuf = new char[rowSize];

		for (unsigned int i = 0; i < height; ++i)
		{
			file.read(reinterpret_cast<char*>(tempBuf), rowSize);

			for (unsigned int j = 0; j < width; ++j)
			{
				unsigned int idx = (height - 1 - i) * width + j;
				imgB.r[idx] = tempBuf[j * 3 + 0];
				imgB.g[idx] = tempBuf[j * 3 + 1];
				imgB.b[idx] = tempBuf[j * 3 + 2];
			}
		}

		delete[] tempBuf;
		
		file.close();
		return true;
	}


	bool StoreRGBs(const char* storeName, const RAWImageBuf_s& imgB)
	{
		bool result = false;

		std::ofstream file(storeName, std::ios::binary);
		if (!file)
		{
			fprintf(stderr, "file failed with error : FILE_ALREADY_EXIST\n");
			return false;
		}

		uint32_t width = imgB.imgW;
		uint32_t height = imgB.imgH;

		unsigned int rowSize = BMPROWSIZE(width);
		unsigned int imageSize = rowSize * height;
		

		BmpFileHeader_s fileHeader;
		fileHeader.fileMarker = BMP_MARKER;
		fileHeader.bufferSize = FILE_MARKER_SIZE + FILE_BODY_SIZE + imageSize;
		fileHeader.imageDataOffet = FILE_MARKER_SIZE + FILE_BODY_SIZE;

		file.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));


		BmpInfoHeader_s infoHeader;
		infoHeader.biSize = FILE_BODY_SIZE;
		infoHeader.biWidth = width;
		infoHeader.biHeight = height;
		infoHeader.biPlanes = 1;
		infoHeader.biBitCount = BIT_PER_PIXEL;
		infoHeader.biCompression= 0; // no compression
		infoHeader.biSizeImage = imageSize;
		infoHeader.biXPelsPerMeter = 0;
		infoHeader.biYPelsPerMeter = 0;
		infoHeader.biClrUsed = 0;
		infoHeader.biClrImportant = 0;

		file.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));


		char* tempBuf = new char[rowSize];

		for (unsigned int i = 0; i < height; ++i)
		{
			for (unsigned int j = 0; j < width; ++j)
			{
				unsigned int idx = (height - 1 - i) * width + j;
				tempBuf[j * 3 + 0] = imgB.b[idx];
				tempBuf[j * 3 + 1] = imgB.g[idx];
				tempBuf[j * 3 + 2] = imgB.r[idx];
			}

			file.write(reinterpret_cast<char*>(tempBuf), rowSize);
		}

		delete[] tempBuf;

		result = true;

	LB_RETURN:

		if (file) { file.close(); }

		return result;
	}
}