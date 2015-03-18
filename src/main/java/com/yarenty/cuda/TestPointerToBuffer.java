package com.yarenty.cuda;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2012 Marco Hutter - http://www.jcuda.org
 */

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * A test and demonstration of the behavior of the
 * {@link Pointer#to(Buffer)} and {@link Pointer#toBuffer(Buffer)} methods.
 */
public class TestPointerToBuffer {
    /**
     * Entry point of this test
     *
     * @param args Not used
     */
    public static void main(String[] args) {
        // Create an array-backed float buffer containing values 0 to 7
        float array[] = {0, 1, 2, 3, 4, 5, 6, 7};
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);

        // Create a direct float buffer containing the same values
        FloatBuffer directBuffer =
                ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                        order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();

        // We're optimistic.
        boolean passed = true;

        // Copy 4 elements of the buffer into an array.
        // The array will contain the first 4 elements.
        System.out.println("\nCopy original buffer");
        passed &= copyWithTo(arrayBuffer, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(arrayBuffer, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithTo(directBuffer, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(directBuffer, 4, new float[]{0, 1, 2, 3});

        // Advance the buffer position, and copy 4 elements
        // into an array. The Pointer#to(Buffer) method will
        // ignore the position, and thus again copy the first
        // 4 elements. The Pointer#toBuffer(Buffer) method
        // will take the position into account, and thus copy
        // the elements 2,3,4,5
        System.out.println("\nCopy buffer with position 2");
        arrayBuffer.position(2);
        directBuffer.position(2);
        passed &= copyWithTo(arrayBuffer, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(arrayBuffer, 4, new float[]{2, 3, 4, 5});
        passed &= copyWithTo(directBuffer, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(directBuffer, 4, new float[]{2, 3, 4, 5});

        // Create a slice of the buffer, and copy 4 elements
        // of the slice into an array. The slice will contain
        // the 6 remaining elements of the buffer: 2,3,4,5,6,7.
        // The Pointer#to method will
        // - ignore slice offset for buffers with backing arrays
        // - consider the slice offset for direct buffers
        // The Pointer#toBuffer method will take the slice offset into
        // account in any case
        System.out.println("\nCopy slice with offset 2");
        FloatBuffer arraySlice = arrayBuffer.slice();
        FloatBuffer directSlice = directBuffer.slice();
        passed &= copyWithTo(arraySlice, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(arraySlice, 4, new float[]{2, 3, 4, 5});
        passed &= copyWithTo(directSlice, 4, new float[]{2, 3, 4, 5});
        passed &= copyWithToBuffer(directSlice, 4, new float[]{2, 3, 4, 5});

        // Set the position of the slice to 2.
        // The Pointer#to method will
        // - ignore slice offset and position for buffers with backing arrays
        // - consider the slice offset, but not the position for direct buffers
        // The Pointer#toBuffer method will take the slice offset
        // and positions into account in any case
        System.out.println("\nCopy slice with offset 2 and position 2");
        arraySlice.position(2);
        directSlice.position(2);
        passed &= copyWithTo(arraySlice, 4, new float[]{0, 1, 2, 3});
        passed &= copyWithToBuffer(arraySlice, 4, new float[]{4, 5, 6, 7});
        passed &= copyWithTo(directSlice, 4, new float[]{2, 3, 4, 5});
        passed &= copyWithToBuffer(directSlice, 4, new float[]{4, 5, 6, 7});

        if (passed) {
            System.out.println("\nPASSED");
        } else {
            System.out.println("\nFAILED");
        }

    }

    /**
     * Copy data from the given buffer into an array, and return whether
     * the array contents matches the expected result. The data will
     * be copied from the given buffer using a Pointer that is created
     * using the {@link Pointer#to(Buffer)} method.
     *
     * @param buffer   The buffer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copyWithTo(
            FloatBuffer buffer, int elements, float[] expected) {
        System.out.println("\nPerforming copy with Pointer#to");
        return copy(buffer, Pointer.to(buffer), elements, expected);
    }

    /**
     * Copy data from the given buffer into an array, and return whether
     * the array contents matches the expected result. The data will
     * be copied from the given buffer using a Pointer that is created
     * using the {@link Pointer#toBuffer(Buffer)} method.
     *
     * @param buffer   The buffer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copyWithToBuffer(
            FloatBuffer buffer, int elements, float[] expected) {
        System.out.println("\nPerforming copy with Pointer#toBuffer");
        return copy(buffer, Pointer.toBuffer(buffer), elements, expected);
    }

    /**
     * Copy data from the given buffer into an array, using the given
     * pointer, and return whether the array contents matches the
     * expected result
     *
     * @param buffer   The buffer
     * @param pointer  The pointer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copy(
            FloatBuffer buffer, Pointer pointer, int elements, float expected[]) {
        System.out.println("Buffer     : " + buffer);
        System.out.println("position   : " + buffer.position());
        System.out.println("limit      : " + buffer.limit());
        if (buffer.hasArray()) {
            System.out.println("arrayOffset: " + buffer.arrayOffset() + " ");
            System.out.println("array      : " +
                    Arrays.toString(buffer.array()));
        }
        System.out.print("contents   : ");
        for (int i = buffer.position(); i < buffer.limit(); i++) {
            System.out.print(buffer.get(i));
            if (i < buffer.limit() - 1) {
                System.out.print(", ");
            }
        }
        System.out.println();

        float result[] = new float[elements];
        JCuda.cudaMemcpy(Pointer.to(result), pointer,
                elements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToHost);

        boolean passed = Arrays.equals(result, expected);
        System.out.println("result     : " + Arrays.toString(result));
        System.out.println("passed?    : " + passed);
        return passed;
    }

}