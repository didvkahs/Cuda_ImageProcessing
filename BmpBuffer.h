#pragma once

struct RAWImageBuf_s
{
	uint32_t imgW = 0;
	uint32_t imgH = 0;

	uint8_t* r = nullptr;
	uint8_t* g = nullptr;
	uint8_t* b = nullptr;
};