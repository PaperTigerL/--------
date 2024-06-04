#include <iostream>
#include <CL/cl.h>
#include <cstring>

#define CL_TARGET_OPENCL_VERSION 120

const char *kernelSource = "__kernel void matrix_mult_kernel(__global const float* A, __global const float* B, __global float* C, int rowsA, int colsA, int colsB) {\n"
"    int globalRow = get_global_id(0);\n"
"    int globalCol = get_global_id(1);\n"
"    float sum = 0.0;\n"
"\n"
"    if (globalRow < rowsA && globalCol < colsB) {\n"
"        for (int i = 0; i < colsA; i++) {\n"
"            sum += A[globalRow * colsA + i] * B[i * colsB + globalCol];\n"
"        }\n"
"        C[globalRow * colsB + globalCol] = sum;\n"
"    }\n"
"}";

void checkOpenCLError(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << ": " << err << std::endl;
        exit(1);
    }
}

int main() {
    const int rowsA = 3;
    const int colsA = 2;
    const int colsB = 4;
    float A[rowsA * colsA];
    float B[colsA * colsB];
    float C[rowsA * colsB];
    float CCPU[rowsA * colsB];

    // Initialize A and B matrices with data
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            A[i * colsA + j] = i * colsA + j + 1; // Initialize matrix A elements
        }
    }

    for (int i = 0; i < colsA; i++) {
        for (int j = 0; j < colsB; j++) {
            B[i * colsB + j] = i * colsB + j + 1; // Initialize matrix B elements
        }
    }

    // Load OpenCL kernel source code
    const char *source = kernelSource;
    size_t sourceSize[] = {strlen(source)};

    // Declare and initialize OpenCL variables for GPU
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id gpuDevice;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpuDevice, NULL);

    cl_context gpuContext = clCreateContext(NULL, 1, &gpuDevice, NULL, NULL, NULL);
    cl_command_queue gpuCommandQueue = clCreateCommandQueue(gpuContext, gpuDevice, 0, NULL);

    // Create OpenCL program and build the kernel for GPU
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, &source, sourceSize, NULL);
    checkOpenCLError(clBuildProgram(gpuProgram, 1, &gpuDevice, NULL, NULL, NULL), "building GPU program");

    // Create kernel object for GPU
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "matrix_mult_kernel", NULL);

    // Allocate memory and set kernel arguments for GPU
    cl_mem inputBufferA = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(float) * rowsA * colsA, NULL, NULL);
    cl_mem inputBufferB = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(float) * colsA * colsB, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, sizeof(float) * rowsA * colsB, NULL, NULL);

    checkOpenCLError(clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &inputBufferA), "setting GPU kernel argument 0");
    checkOpenCLError(clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &inputBufferB), "setting GPU kernel argument 1");
    checkOpenCLError(clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &outputBuffer), "setting GPU kernel argument 2");
    checkOpenCLError(clSetKernelArg(gpuKernel, 3, sizeof(int), &rowsA), "setting GPU kernel argument 3");
    checkOpenCLError(clSetKernelArg(gpuKernel, 4, sizeof(int), &colsA), "setting GPU kernel argument 4");
    checkOpenCLError(clSetKernelArg(gpuKernel, 5, sizeof(int), &colsB), "setting GPU kernel argument 5");

    // Transfer data to GPU
    checkOpenCLError(clEnqueueWriteBuffer(gpuCommandQueue, inputBufferA, CL_TRUE, 0, sizeof(float) * rowsA * colsA, A, 0, NULL, NULL), "writing GPU buffer A");
    checkOpenCLError(clEnqueueWriteBuffer(gpuCommandQueue, inputBufferB, CL_TRUE, 0, sizeof(float) * colsA * colsB, B, 0, NULL, NULL), "writing GPU buffer B");

    // Execute the kernel on GPU
    size_t globalWorkSize[2] = {colsB, rowsA};
    checkOpenCLError(clEnqueueNDRangeKernel(gpuCommandQueue, gpuKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL), "enqueueing GPU kernel");
    checkOpenCLError(clFinish(gpuCommandQueue), "waiting for GPU kernel to finish");

    // Read the result from GPU
    checkOpenCLError(clEnqueueReadBuffer(gpuCommandQueue, outputBuffer, CL_TRUE, 0, sizeof(float) * rowsA * colsB, C, 0, NULL, NULL), "reading GPU buffer");

    // Release OpenCL resources for GPU
    clReleaseMemObject(inputBufferA);
    clReleaseMemObject(inputBufferB);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(gpuKernel);
    clReleaseProgram(gpuProgram);
    clReleaseCommandQueue(gpuCommandQueue);
    clReleaseContext(gpuContext);

    // Output the result matrix C from GPU
    std::cout << "Result from GPU: " << std::endl;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << C[i * colsB + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}