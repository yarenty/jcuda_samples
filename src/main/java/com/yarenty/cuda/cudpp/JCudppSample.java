package com.yarenty.cuda.cudpp;

/*
 * JCudpp - Java bindings for CUDPP, the CUDA Data Parallel
 * Primitives Library, to be used with JCuda<br />
 * http://www.jcuda.org
 *
 * Copyright 2009 Marco Hutter - http://www.jcuda.org
 */

import java.util.Random;

//import jcuda.jcudpp.*;

/**
 * This is a sample class demonstrating the application of JCudpp for
 * performing a sort of an integer array with 1000000 elements.
 */
public class JCudppSample {
    public static void main(final String args[]) {
        //       testSort(1000000);
    }

    /**
     * Test the JCudpp sort operation for an array of size n
     *
     * @param n The array size
     */
 /*   public static boolean testSort(int N)
    {
        System.out.println("Creating input data");
        int array[] = createRandomIntData(N);
        int arrayRef[] = array.clone();

        System.out.println("Performing sort with Java...");
        Arrays.sort(arrayRef);

        System.out.println("Performing sort with JCudpp...");
        sort(array);

        boolean passed = Arrays.equals(array, arrayRef);
        System.out.println("testSort "+(passed?"PASSED":"FAILED"));
        return passed;
    }

    /**
     * Implementation of sort using JCudpp
     *
     * @param array The array to sort
     */
 /*   private static void sort(int array[])
    {
        int n = array.length;

        // Allocate memory on the device
        Pointer d_keys = new Pointer();
        JCuda.cudaMalloc(d_keys, n * Sizeof.INT);

        // Copy the input array from the host to the device
        JCuda.cudaMemcpy(d_keys, Pointer.to(array), n * Sizeof.INT,
                cudaMemcpyKind.cudaMemcpyHostToDevice);

        // Create a CUDPPConfiguration for a radix sort of
        // an array of ints
        CUDPPConfiguration config = new CUDPPConfiguration();
        config.algorithm = CUDPPAlgorithm.CUDPP_SORT_RADIX;
        config.datatype = CUDPPDatatype.CUDPP_UINT;
        config.op = CUDPPOperator.CUDPP_ADD;
        config.options = CUDPPOption.CUDPP_OPTION_KEYS_ONLY;

        // Create a CUDPPHandle for the sort operation
        CUDPPHandle theCudpp = new CUDPPHandle();
        JCudpp.cudppCreate(theCudpp);
        CUDPPHandle handle = new CUDPPHandle();
        JCudpp.cudppPlan(theCudpp, handle, config, n, 1, 0);

        // Execute the sort operation
        JCudpp.cudppSort(handle, d_keys, null, n);

        Arrays.fill(array, 0);

        // Copy the result from the device to the host
        JCuda.cudaMemcpy(Pointer.to(array), d_keys, n * Sizeof.INT,
                cudaMemcpyKind.cudaMemcpyDeviceToHost);

        // Clean up
        JCudpp.cudppDestroyPlan(handle);
        JCudpp.cudppDestroy(theCudpp);
        JCuda.cudaFree(d_keys);

    }

    /**
     * Creates an array of the specified size, containing some random data
     */
    private static int[] createRandomIntData(final int n) {
        final Random random = new Random(0);
        final int x[] = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = random.nextInt(10);
        }
        return x;
    }


}