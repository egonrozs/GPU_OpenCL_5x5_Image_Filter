// OCLTest1.cpp : Defines the entry point for the console application.
//

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>

#include "time.h"

#include "CL\cl.h"

#include "defs.h"
#include "func.h"

////////////////////////////////////////////////////
// SEE defs.h for kernel selection!!!


float filter_laplace_f[5][5] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
-1.0f, -1.0f, 24.0f, -1.0f, -1.0f,
-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
-1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

int filter_laplace[5][5] = { -1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 24, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1 };

const char *getErrorString(cl_int error)
{
	switch (error){
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

#define MAX_PROG_SIZE 65536

void conv_filter_ocl(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
	int imgFOfssetH, int imgFOfssetW,
	unsigned char *imgSrc, unsigned char *imgDst)
{
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	clock_t s0, e0;
	double d0;

	int size_in;
	size_in = imgHeightF*imgWidthF * 3;
	int size_out;
	size_out = imgHeight*imgWidth * 3;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Init OpenCL

	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms;
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)* ret_num_platforms);
	ret = clGetPlatformIDs(ret_num_platforms, platforms, &ret_num_platforms);
	
	int num_devices_all = 0;
	for (int platform_id = 0; platform_id < ret_num_platforms; platform_id++)
	{
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
		num_devices_all = num_devices_all + ret_num_devices;
	}
	cl_device_id *devices;
	int device_offset = 0;
	devices = (cl_device_id*)malloc(sizeof(cl_device_id)* num_devices_all);
	for (int platform_id = 0; platform_id < ret_num_platforms; platform_id++)
	{
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, ret_num_devices, &devices[device_offset], &ret_num_devices);
		device_offset = device_offset + ret_num_devices;
	}

	char cBuffer[1024];
	for (int device_num = 0; device_num < num_devices_all; device_num++)
	{
		printf("Device id: %d,  ", device_num);

		ret = clGetDeviceInfo(devices[device_num], CL_DEVICE_VENDOR, sizeof(cBuffer), &cBuffer, NULL);
		printf("%s ", cBuffer);

		ret = clGetDeviceInfo(devices[device_num], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		printf("%s\r\n", cBuffer);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Select device to be used
#if FIXED_OCL_DEVICE == 0
	printf("\n\nSelect OpenCL device and press enter:");
	int device_sel = getchar()-0x30;
	device_id = devices[device_sel];
#else	
	device_id = devices[FIXED_OCL_DEVICE_ID];
	free(devices);
#endif

	// Load the source code containing the kernel
	char *kernel_source;
	size_t kernel_size;
	
	FILE *kernel_file;
	char fileName[] = KERNEL_FILE_NAME;

	fopen_s(&kernel_file, fileName, "r");
	if (kernel_file == NULL) {
		fprintf(stderr, "Failed to read kernel from file.\n");
		exit(1);
	}
	fseek(kernel_file, 0, SEEK_END);
	kernel_size = ftell(kernel_file);
	rewind(kernel_file);
	kernel_source = (char *)malloc(kernel_size + 1);
	kernel_source[kernel_size] = '\0';
	int read = fread(kernel_source, sizeof(char), kernel_size, kernel_file);
	if (read != kernel_size) {
		fprintf(stderr, "Error while reading the kernel.\n");
		exit(1);
	}
	fclose(kernel_file);

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

	/* Create Memory Buffer on device*/
	cl_mem device_imgSrc, device_imgDst, device_coeffs;
	device_imgSrc = clCreateBuffer(context, CL_MEM_READ_ONLY, size_in, NULL, &ret);
	device_imgDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_out, NULL, &ret);
#if USE_FLOAT_COEFFS == 1
	device_coeffs = clCreateBuffer(context, CL_MEM_READ_ONLY, 5*5*sizeof(float), NULL, &ret);
#else
	device_coeffs = clCreateBuffer(context, CL_MEM_READ_ONLY, 5 * 5 * sizeof(int), NULL, &ret);
#endif

	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source,
		(const size_t *)&kernel_size, &ret);

	/* Build Kernel Program */
	//ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	ret = clBuildProgram(program, 1, &device_id, "-cl-nv-verbose", NULL, NULL);
	size_t param_value_size, param_value_size_ret;



	if (ret != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ %s ]\n", ret, getErrorString(ret));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void*)&bldstatus, &len);
		printf("Build Status %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", getErrorString(bldstatus));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		printf("Build Options %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("Build Log %d: %s\n", ret, getErrorString(ret));
		printf("%s\n", buffer);
		exit(1);
	}
	else
	{
		size_t len;
		char buffer[2048];
		cl_build_status bldstatus;
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("Build Log %d: %s\n", ret, getErrorString(ret));
		printf("%s\n", buffer);
	}

	if (ret != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ %s ]\n", ret, getErrorString(ret));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
		printf("Build Status %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", getErrorString(bldstatus));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		printf("Build Options %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("Build Log %d: %s\n", ret, getErrorString(ret));
		printf("%s\n", buffer);
		exit(1);
	}


	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, KERNEL_FUNCTION, &ret);

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(device_imgSrc), (void *)&device_imgSrc);
	ret = clSetKernelArg(kernel, 1, sizeof(device_imgDst), (void *)&device_imgDst);
	ret = clSetKernelArg(kernel, 2, sizeof(device_coeffs), (void *)&device_coeffs);
	ret = clSetKernelArg(kernel, 3, sizeof(int), &imgWidth);
	ret = clSetKernelArg(kernel, 4, sizeof(int), &imgWidthF);


	// Copy input data to device memory
	ret = clEnqueueWriteBuffer(command_queue, device_imgSrc, CL_TRUE, 0,
		size_in, imgSrc, 0, NULL, NULL);
#if USE_FLOAT_COEFFS == 1
	ret = clEnqueueWriteBuffer(command_queue, device_coeffs, CL_TRUE, 0,
		25 * sizeof(float), filter_laplace_f, 0, NULL, NULL);
#else
	ret = clEnqueueWriteBuffer(command_queue, device_coeffs, CL_TRUE, 0,
		25 * sizeof(int), filter_laplace, 0, NULL, NULL);
#endif

	clFinish(command_queue);
	
	/* Execute OpenCL Kernel */    //Ezekkel a paraméterekkel lehet változtatni a szálakat
	int imgHeight_BMS2 = imgHeight/4; //eredetileg nincs benne 2v4 el kell osztani
	size_t local_size[] = { LOCAL_SIZE_X, LOCAL_SIZE_Y };
	//size_t global_size[] = { imgWidth, imgHeight }; 
	size_t global_size[] = { imgWidth, imgHeight_BMS2 }; //BMS2 esetén ezt kell bennhagyni

	time_measure(1);

	cl_event event[1024];
	for (int runs = 0; runs < KERNEL_RUNS; runs++)
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event[runs]);

	if (ret != CL_SUCCESS)
	{
		printf("\nError %d: Failed to build program executable [ %s ]\n", ret, getErrorString(ret));
		exit(1);
	}

	clWaitForEvents(1, &event[KERNEL_RUNS - 1]);

	double runtime = time_measure(2);

	cl_ulong time_start, time_end;
	double total_time;
	clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event[KERNEL_RUNS-1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("Total kernel time = %6.4f ms, # of runs: %d\r\n", (total_time / (1000000.0)), KERNEL_RUNS);
	double mpixel = (KERNEL_RUNS * 1000.0 * double(imgWidth*imgHeight) / (total_time / (1000000.0))) / 1000000;
	printf("Single run MPixel/s: %4.4f\r\n", mpixel);
	printf("Meas time: %6.4f ms\r\n", (total_time/1000000.0));



	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, device_imgDst, CL_TRUE, 0,
		size_out, imgDst, 0, NULL, NULL);


	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(kernel_source);

}
