// lab1.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include "memory.h"
#include "time.h"

#include "omp.h"

#include <IL/ilut.h>
#include <IL/ilu.h>

#include "emmintrin.h"
#include "nmmintrin.h"

#include "defs.h"
#include "func.h"


int main()
{
#if USE_DEVIL == 1
	ilInit(); iluInit();
	ILboolean ret;
	ILuint ilImg=0;
	ilGenImages(1, &ilImg);
	ilBindImage(ilImg);
    ret = ilLoadImage((const wchar_t*)("input.jpg"));
	ILubyte* imgData = ilGetData(); 

	int imgWidth = ilGetInteger(IL_IMAGE_WIDTH);
	int imgHeight = ilGetInteger(IL_IMAGE_HEIGHT);
	ILint imgOrigin = ilGetInteger(IL_ORIGIN_MODE);
#else
	FILE *raw_in_file;
	raw_in_file = fopen("input.raw", "rb");
	if (raw_in_file == NULL)
	{
		printf("Error opening raw file \r\n");
		exit(-1);
	}
	
	int imgWidth = 4992;
	int imgHeight = 3744;

	unsigned char *imgData;
	imgData = (unsigned char *)malloc(imgWidth * imgHeight * 3 * sizeof(unsigned char));

	fread(imgData, 1, imgWidth*imgHeight * 3, raw_in_file);

	fclose(raw_in_file);
#endif


	printf("Input resolution: %4dx%4d\n", imgWidth, imgHeight);

	unsigned char *imgSrc;
	int imgWidthF = imgWidth+FILTER_W-1;
	int imgHeightF = imgHeight+FILTER_H-1;
	int imgFOfssetW = (FILTER_W-1)/2;
	int imgFOfssetH = (FILTER_H-1)/2;
	imgSrc   = (unsigned char *)(_aligned_malloc(3*imgWidthF*imgHeightF*sizeof(unsigned char), 32));
    int row, col;

	for (row=0; row<imgHeightF; row++)
	{
		for (col=0; col<imgWidthF;col++)
		{
			int pixel = (row*imgWidthF + col)*3;
			*(imgSrc + pixel + 0) = 0;
			*(imgSrc + pixel + 1) = 0;
			*(imgSrc + pixel + 2) = 0;
		}
	}

	for (row=0; row<imgHeight; row++)
	{
		for (col=0; col<imgWidth;col++)
		{
			int pixel_dst = ((row+imgFOfssetH)*imgWidthF + (col+imgFOfssetW))*3;
			int pixel_src = (row*imgWidth + col)*3;
			*(imgSrc + pixel_dst + 0) = (unsigned char)(*(imgData + pixel_src + 0));
			*(imgSrc + pixel_dst + 1) = (unsigned char)(*(imgData + pixel_src + 1));
			*(imgSrc + pixel_dst + 2) = (unsigned char)(*(imgData + pixel_src + 2));
		}
	}


// IMAGE PROCESSING
//---------------------------------------------------------------------------------------
	clock_t s0, e0;
    double d0;

	unsigned char *imgRes;
    imgRes = (unsigned char *)(_aligned_malloc(3*imgWidth*imgHeight*sizeof(unsigned char), 32));

	float filter_laplace[] = {-1.0, -1.0, -1.0, -1.0, -1.0,
		                      -1.0, -1.0, -1.0, -1.0, -1.0,
		                      -1.0, -1.0, 24.0, -1.0, -1.0,
		                      -1.0, -1.0, -1.0, -1.0, -1.0,
		                      -1.0, -1.0, -1.0, -1.0, -1.0};
	double mpixel;

#if 0
	s0 = clock();
	for (int r=0; r<RUNS; r++)
	{
		conv_filter(imgHeight, imgWidth, imgHeightF, imgWidthF,
					imgFOfssetH, imgFOfssetW,
					filter_laplace, imgSrc, imgRes);
	}

    e0 = clock();
    d0 = (double)(e0-s0)/(RUNS*CLOCKS_PER_SEC);
	mpixel = (imgWidth*imgHeight/d0)/1000000;
    printf("C CPU TIME: %4.4f\n", d0);
	printf("C Mpixel/s: %4.4f\n", mpixel);
#endif


#if 1
	conv_filter_ocl(imgHeight, imgWidth, imgHeightF, imgWidthF,
		imgFOfssetH, imgFOfssetW,
		imgSrc, imgRes);
#endif

//---------------------------------------------------------------------------------------
// IMAGE PROCESSING END

#if USE_DEVIL == 1
	for (row=0; row<imgHeight; row++)
	{
		for (col=0; col<imgWidth;col++)
		{
			int pixel_src = (row*imgWidth + col) * 3;
			int pixel_dst = (row*imgWidth + col) * 3;
			*(imgData + pixel_dst + 0) = (ILubyte)(*(imgRes + pixel_src + 0));
			*(imgData + pixel_dst + 1) = (ILubyte)(*(imgRes + pixel_src + 1));
			*(imgData + pixel_dst + 2) = (ILubyte)(*(imgRes + pixel_src + 2));
		}
	}
	ret = ilSetData(imgData);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage((const wchar_t*)("output.jpg"));
	ilDeleteImages(1, &ilImg);
#else
	for (row = 0; row<imgHeight; row++)
	{
		for (col = 0; col<imgWidth; col++)
		{
			int pixel_src = ((row + imgFOfssetH)*imgWidthF + (col + imgFOfssetW)) * 3;
			int pixel_dst = (row*imgWidth + col) * 3;
			*(imgData + pixel_dst + 0) = (ILubyte)(*(imgRes + pixel_src + 0));
			*(imgData + pixel_dst + 1) = (ILubyte)(*(imgRes + pixel_src + 1));
			*(imgData + pixel_dst + 2) = (ILubyte)(*(imgRes + pixel_src + 2));
		}
	}

	FILE *raw_out_file;
	raw_out_file = fopen("output.raw", "wb");
	if (raw_out_file == NULL)
	{
		printf("Error opening raw output file \r\n");
		exit(-1);
	}

	fwrite(imgData, sizeof(unsigned char), imgWidth*imgHeight * 3, raw_out_file);

	fclose(raw_out_file);
#endif

	_aligned_free(imgSrc);
    _aligned_free(imgRes);

	return 0;
}

