#define FILTER_W 5
#define FILTER_H 5

#define RUNS 1 //feltehetõleg a cpuhoz van
#define KERNEL_RUNS 1000 //Kernel futás

#define USE_DEVIL 1

#define FIXED_OCL_DEVICE 0
#define FIXED_OCL_DEVICE_ID 0

#define KERNEL_FILE_NAME ".\\_src\\opencl_kernels.cl"
#define KERNEL_FUNCTION "median_filter_BMS2_shared_32x8" //ezt kell a meghívandó névre átírni a kernel híváshoz saját kódot 32*32-re kell állítani az is gyorsít
#define LOCAL_SIZE_X 32 //16 v 32						// workgroup X size: 16 for all kernels, except the last (32)
#define LOCAL_SIZE_Y 8 //8 v 16 v 32                   // workgroup X size: 16 for all kernels, except the last (8), see kernel codes
#define USE_FLOAT_COEFFS 1   // 0: use int coeffs, 1: use float coeffs //ha float van akkor ezt be kell billenteni


#define OCLGRIND_INTERACTIVE = 1