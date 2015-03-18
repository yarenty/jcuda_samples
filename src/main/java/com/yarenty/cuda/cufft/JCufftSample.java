package com.yarenty.cuda.cufft;

/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
 * to be used with JCuda<br />
 * http://www.jcuda.org
 *
 * Copyright 2009 Marco Hutter - http://www.jcuda.org
 */

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;

import java.util.Random;

/**
 * This is a sample class that performs a 1D Complex-To-Complex
 * forward FFT with JCufft, and compares the result to the
 * reference computed with JTransforms.
 */
class JCufftSample
{
    public static void main(String args[])
    {
        testC2C1D(1<<20);
    }

    /**
     * Test the 1D C2C transform with the given size.
     *
     * @param size The size of the transform
     */
    private static void testC2C1D(int size)
    {
        System.out.println("Creating input data...");
        float input[] = createRandomFloatData(size * 2);

        System.out.println("Performing 1D C2C transform with JTransforms...");
        float outputJTransforms[] = input.clone();
        FloatFFT_1D fft = new FloatFFT_1D(size);
        fft.complexForward(outputJTransforms);

        System.out.println("Performing 1D C2C transform with JCufft...");
        float outputJCufft[] = input.clone();
        cufftHandle plan = new cufftHandle();
        JCufft.cufftPlan1d(plan, size, cufftType.CUFFT_C2C, 1);
        JCufft.cufftExecC2C(plan, outputJCufft, outputJCufft, JCufft.CUFFT_FORWARD);
        JCufft.cufftDestroy(plan);

        boolean passed = isCorrectResult(outputJTransforms, outputJCufft);
        System.out.println("testC2C1D "+(passed?"PASSED":"FAILED"));
    }

    /**
     * Creates an array of the specified size, containing some random data
     */
    private static float[] createRandomFloatData(int x)
    {
        Random random = new Random(1);
        float a[] = new float[x];
        for (int i=0; i<x; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }

    /**
     * Compares the given result against a reference, and returns whether the
     * error norm is below a small epsilon threshold
     */
    private static boolean isCorrectResult(float result[], float reference[])
    {
        float errorNorm = 0;
        float refNorm = 0;
        for (int i = 0; i < result.length; ++i)
        {
            float diff = reference[i] - result[i];
            errorNorm += diff * diff;
            refNorm += reference[i] * result[i];
        }
        errorNorm = (float) Math.sqrt(errorNorm);
        refNorm = (float) Math.sqrt(refNorm);
        if (Math.abs(refNorm) < 1e-6)
        {
            return false;
        }
        return (errorNorm / refNorm < 1e-6f);
    }


}