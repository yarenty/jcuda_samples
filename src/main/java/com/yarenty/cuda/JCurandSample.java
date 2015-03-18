package com.yarenty.cuda;

/*
 * JCurand - Java bindings for CURAND, the NVIDIA CUDA random
 * number generation library, to be used with JCuda
 *
 * Copyright (c) 2010 Marco Hutter - http://www.jcuda.org
 */

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;

import java.util.Arrays;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

/**
 * A small sample application showing how to use JCurand.<br />
 * <br />
 * Basically ported from the NVIDIA CURAND documentation example.
 */
public class JCurandSample {
    public static void main(String args[]) {
        // Enable exceptions and omit all subsequent error checks
        JCuda.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);

        int n = 100;
        curandGenerator generator = new curandGenerator();

        // Allocate n floats on host
        float hostData[] = new float[n];

        // Allocate n floats on device
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);

        // Create pseudo-random number generator
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed
        curandSetPseudoRandomGeneratorSeed(generator, 1234);

        // Generate n floats on device
        curandGenerateUniform(generator, deviceData, n);

        // Copy device memory to host
        cudaMemcpy(Pointer.to(hostData), deviceData,
                n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        // Show result
        System.out.println(Arrays.toString(hostData));

        // Cleanup
        curandDestroyGenerator(generator);
        cudaFree(deviceData);
    }
}