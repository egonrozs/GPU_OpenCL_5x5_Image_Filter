double time_measure(int mode);

void conv_filter(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
				 int imgFOfssetH, int imgFOfssetW,
				 float *filter, unsigned char *imgSrc, unsigned char *imgDst);

void conv_filter_ocl(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
	int imgFOfssetH, int imgFOfssetW,
	unsigned char *imgSrc, unsigned char *imgDst);