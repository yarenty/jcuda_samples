package com.yarenty.mix;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2010 Marco Hutter - http://www.jcuda.org
 */

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.jcublas.JCublas;
import jcuda.runtime.cudaMemcpyKind;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.JCublas.cublasInit;
import static jcuda.jcublas.JCublas.cublasSnrm2;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;

/**
 * This is a simple example that shows how the interoperability between
 * the CUDA runtime- and driver API may be used with JCuda. <br />
 * <br />
 * The example creates a vector on the device using the runtime API,
 * computes the norm of a vector using JCublas, then inverts all
 * elements of the vector using a kernel that is executed with the
 * driver API, computes the norm of the resulting vector with JCublas,
 * and finally copies the vector back using the driver API.
 */
public class JCudaRuntimeDriverMixSample {

    public static void main(final String args[]) throws IOException {
        JCudaDriver.setExceptionsEnabled(true);
        JCublas.setExceptionsEnabled(true);

        final String ptxFileName = preparePtxFile("invertVectorElements.cu");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        final CUcontext context = new CUcontext();
        final CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the PTX file and obtain the "invertVectorElements" function.
        final CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        final CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "invertVectorElements");

        // Create the input data.
        final int n = 5;
        final Random random = new Random(0);
        final float vector[] = new float[n];
        for (int i = 0; i < n; i++) {
            vector[i] = random.nextFloat();
        }

        // Copy the vector to the device using the Runtime API
        final CUdeviceptr vectorDevice = new CUdeviceptr();
        cudaMalloc(vectorDevice, n * 2 * Sizeof.FLOAT);
        cudaMemcpy(vectorDevice, Pointer.to(vector), n * 2 * Sizeof.FLOAT,
                cudaMemcpyKind.cudaMemcpyHostToDevice);

        // Use JCublas to compute the vector norm
        cublasInit();
        final float norm = cublasSnrm2(n, vectorDevice, 1);

        System.out.println("Input vector    " + Arrays.toString(vector));
        System.out.println("Norm            " + norm);

        // Call the kernel function.
        final Pointer kernelParameters = Pointer.to(
                Pointer.to(vectorDevice),
                Pointer.to(new int[]{n})
        );
        final int blockX = n;
        final int gridX = 1;
        cuLaunchKernel(function,
                gridX, 1, 1,           // Grid dimension
                blockX, 1, 1,          // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        cuCtxSynchronize();

        // Use JCublas to compute the norm of the vector that
        // was modified using the kernel
        final float invNorm = cublasSnrm2(n, vectorDevice, 1);

        // Copy the vector back to the host using the Driver API
        cuMemcpyDtoH(Pointer.to(vector), vectorDevice, n * 2 * Sizeof.FLOAT);

        // Print the results
        System.out.println("Inverted vector " + Arrays.toString(vector));
        System.out.println("Norm            " + invNorm);

        // Clean up
        cuMemFree(vectorDevice);

        System.out.println("PASSED");
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(final String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        final String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        final File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        final File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        final String modelString = "-m" + System.getProperty("sun.arch.data.model");
        final String command =
                "nvcc " + modelString + " -ptx " +
                        cuFile.getPath() + " -o " + ptxFileName;

        System.out.println("Executing\n" + command);
        final Process process = Runtime.getRuntime().exec(command);

        final String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        final String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(final InputStream inputStream)
            throws IOException {
        final ByteArrayOutputStream baos = new ByteArrayOutputStream();
        final byte buffer[] = new byte[8192];
        while (true) {
            final int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

}
