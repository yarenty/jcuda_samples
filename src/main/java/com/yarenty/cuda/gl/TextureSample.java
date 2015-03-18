package com.yarenty.cuda.gl;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */

import com.jogamp.opengl.util.Animator;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.dim3;

import javax.media.opengl.awt.GLCanvas;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.io.FileInputStream;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

/**
 * A sample illustrating how to use textures with JCuda. This program uses
 * the CUBIN file that is created by the "volumeRender" program from the
 * NVIDIA CUDA samples web site. <br />
 * <br />
 * The program loads an 8 bit RAW volume data set and copies it into a
 * 3D texture. The texture is accessed by the kernel to render an image
 * of the volume data. The resulting image is written into a pixel
 * buffer object (PBO) which is then displayed using JOGL.
 */
public class TextureSample implements GLEventListener {
    /**
     * Entry point for this sample.
     *
     * @param args not used
     */
    public static void main(final String args[]) {
        startSample("Bucky.raw", 32, 32, 32);

        // Other input files may be obtained from http://www.volvis.org
        //startSample("mri_ventricles.raw", 256, 256, 124);
        //startSample("backpack8.raw", 512, 512, 373);
        //startSample("foot.raw", 256, 256, 256);
    }

    /**
     * Starts this sample with the data that is read from the file
     * with the given name. The data is assumed to have the
     * specified dimensions.
     *
     * @param fileName The name of the volume data file to load
     * @param sizeX    The size of the data set in X direction
     * @param sizeY    The size of the data set in Y direction
     * @param sizeZ    The size of the data set in Z direction
     */
    private static void startSample(
            final String fileName, final int sizeX, final int sizeY, final int sizeZ) {
        // Try to read the specified file
        byte data[] = null;
        try {
            final int size = sizeX * sizeY * sizeZ;
            final FileInputStream fis = new FileInputStream(fileName);
            data = new byte[size];
            fis.read(data);
        } catch (IOException e) {
            System.err.println("Could not load input file");
            e.printStackTrace();
            return;
        }

        // Start the sample with the data that was read from the file
        final byte volumeData[] = data;
        final GLProfile profile = GLProfile.get(GLProfile.GL3bc);
        final GLCapabilities capabilities = new GLCapabilities(profile);
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new JCudaDriverTextureSample(capabilities,
                        volumeData, sizeX, sizeY, sizeZ);
            }
        });
    }

    /**
     * The GL component which is used for rendering
     */
    private GLCanvas glComponent;

    /**
     * The animator used for rendering
     */
    private Animator animator;

    /**
     * The CUDA module containing the kernel
     */
    private final CUmodule module = new CUmodule();

    /**
     * The handle for the CUDA function of the kernel that is to be called
     */
    private CUfunction function;

    /**
     * The width of the rendered area and the PBO
     */
    private int width = 0;

    /**
     * The height of the rendered area and the PBO
     */
    private int height = 0;

    /**
     * The size of the volume data that is to be rendered
     */
    private final dim3 volumeSize = new dim3();

    /**
     * The volume data that is to be rendered
     */
    private byte h_volume[];

    /**
     * The block size for the kernel execution
     */
    private final dim3 blockSize = new dim3(16, 16, 1);

    /**
     * The grid size of the kernel execution
     */
    private dim3 gridSize =
            new dim3(width / blockSize.x, height / blockSize.y, 1);

    /**
     * The global variable of the module which stores the
     * inverted view matrix.
     */
    private final CUdeviceptr c_invViewMatrix = new CUdeviceptr();

    /**
     * The inverted view matrix, which will be copied to the global
     * variable of the kernel.
     */
    private final float invViewMatrix[] = new float[12];

    /**
     * The density of the rendered volume data
     */
    private float density = 0.05f;

    /**
     * The brightness of the rendered volume data
     */
    private float brightness = 1.0f;

    /**
     * The transferOffset of the rendered volume data
     */
    private float transferOffset = 0.0f;

    /**
     * The transferScale of the rendered volume data
     */
    private float transferScale = 1.0f;

    /**
     * The OpenGL pixel buffer object
     */
    private int pbo = 0;

    /**
     * The 3D texture reference
     */
    private final CUtexref tex = new CUtexref();

    /**
     * The 1D transfer texture reference
     */
    private final CUtexref transferTex = new CUtexref();

    /**
     * The translation in X-direction
     */
    private float translationX = 0;

    /**
     * The translation in Y-direction
     */
    private float translationY = 0;

    /**
     * The translation in Z-direction
     */
    private float translationZ = -4;

    /**
     * The rotation about the X-axis, in degrees
     */
    private float rotationX = 0;

    /**
     * The rotation about the Y-axis, in degrees
     */
    private float rotationY = 0;

    /**
     * Step counter for FPS computation
     */
    private int step = 0;

    /**
     * Time stamp for FPS computation
     */
    private long prevTimeNS = -1;

    /**
     * The main frame of the application
     */
    private Frame frame;

    /**
     * Inner class encapsulating the MouseMotionListener and
     * MouseWheelListener for the interaction
     */
    class MouseControl implements MouseMotionListener, MouseWheelListener {
        private Point previousMousePosition = new Point();

        @Override
        public void mouseDragged(MouseEvent e) {
            int dx = e.getX() - previousMousePosition.x;
            int dy = e.getY() - previousMousePosition.y;

            // If the left button is held down, move the object
            if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) ==
                    MouseEvent.BUTTON1_DOWN_MASK) {
                translationX += dx / 100.0f;
                translationY -= dy / 100.0f;
            }

            // If the right button is held down, rotate the object
            else if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) ==
                    MouseEvent.BUTTON3_DOWN_MASK) {
                rotationX += dy;
                rotationY += dx;
            }
            previousMousePosition = e.getPoint();
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            previousMousePosition = e.getPoint();
        }

        @Override
        public void mouseWheelMoved(MouseWheelEvent e) {
            // Translate along the Z-axis
            translationZ += e.getWheelRotation() * 0.25f;
            previousMousePosition = e.getPoint();
        }
    }

    /**
     * Creates a new JCudaTextureSample that displays the given volume
     * data, which has the specified size.
     *
     * @param volumeData The volume data
     * @param sizeX      The size of the data set in X direction
     * @param sizeY      The size of the data set in Y direction
     * @param sizeZ      The size of the data set in Z direction
     */
    public TextureSample(final GLCapabilities capabilities,
                         final byte volumeData[], final int sizeX, final int sizeY, final int sizeZ) {
        h_volume = volumeData;
        volumeSize.x = sizeX;
        volumeSize.y = sizeY;
        volumeSize.z = sizeZ;

        width = 800;
        height = 800;

        // Initialize the GL component
        glComponent = new GLCanvas(capabilities);
        glComponent.addGLEventListener(this);

        // Initialize the mouse controls
        final MouseControl mouseControl = new MouseControl();
        glComponent.addMouseMotionListener(mouseControl);
        glComponent.addMouseWheelListener(mouseControl);

        // Create the main frame
        frame = new JFrame("JCuda 3D texture volume rendering sample");
        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                runExit();
            }
        });
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(width, height));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.add(createControlPanel(), BorderLayout.SOUTH);
        frame.pack();
        frame.setVisible(true);

        // Create and start the animator
        animator = new Animator(glComponent);
        animator.setRunAsFastAsPossible(true);
        animator.start();
    }

    /**
     * Create the control panel containing the sliders for setting
     * the visualization parameters.
     *
     * @return The control panel
     */
    private JPanel createControlPanel() {
        final JPanel controlPanel = new JPanel(new GridLayout(2, 2));
        JPanel panel = null;
        JSlider slider = null;

        // Density
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Density:"));
        slider = new JSlider(0, 100, 5);
        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                final JSlider source = (JSlider) e.getSource();
                final float a = source.getValue() / 100.0f;
                density = a;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Brightness
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Brightness:"));
        slider = new JSlider(0, 100, 10);
        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                final JSlider source = (JSlider) e.getSource();
                final float a = source.getValue() / 100.0f;
                brightness = a * 10;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Transfer offset
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Transfer Offset:"));
        slider = new JSlider(0, 100, 55);
        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                final JSlider source = (JSlider) e.getSource();
                final float a = source.getValue() / 100.0f;
                transferOffset = (-0.5f + a) * 2;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Transfer scale
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Transfer Scale:"));
        slider = new JSlider(0, 100, 10);
        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                final JSlider source = (JSlider) e.getSource();
                final float a = source.getValue() / 100.0f;
                transferScale = a * 10;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        return controlPanel;
    }

    /**
     * Implementation of GLEventListener: Called to initialize the
     * GLAutoDrawable. This method will initialize the JCudaDriver
     * and cause the initialization of CUDA and the OpenGL PBO.
     */
    public void init(final GLAutoDrawable drawable) {
        // Perform the default GL initialization
        final GL gl = drawable.getGL();
        gl.setSwapInterval(0);
        gl.glEnable(GL.GL_DEPTH_TEST);
        gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        setupView(drawable);

        // Initialize CUDA with the current volume data
        initCuda();

        // Initialize the OpenGL pixel buffer object
        initPBO(gl);
    }

    /**
     * Initialize CUDA and the 3D texture with the current volume data.
     */
    void initCuda() {
        // Initialize the JCudaDriver. Note that this has to be done from
        // the same thread that will later use the JCudaDriver API.
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        final CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        final CUcontext glCtx = new CUcontext();
        cuGLCtxCreate(glCtx, 0, dev);

        // Load the CUBIN file containing the kernel
        cuModuleLoad(module, "volumeRender_kernel.sm_10.cubin");

        // Obtain the global pointer to the inverted view matrix from
        // the module
        cuModuleGetGlobal(c_invViewMatrix, new long[1], module,
                "c_invViewMatrix");

        // Obtain a function pointer to the kernel function. This function
        // will later be called in the display method of this
        // GLEventListener.
        function = new CUfunction();
        cuModuleGetFunction(function, module,
                "_Z8d_renderPjjjffff");


        // Initialize the data for the transfer function and the volume data
        final CUarray d_transferFuncArray = new CUarray();
        final CUarray d_volumeArray = new CUarray();

        // Create the 3D array that will contain the volume data
        // and will be accessed via the 3D texture
        final CUDA_ARRAY3D_DESCRIPTOR allocateArray = new CUDA_ARRAY3D_DESCRIPTOR();
        allocateArray.Width = volumeSize.x;
        allocateArray.Height = volumeSize.y;
        allocateArray.Depth = volumeSize.z;
        allocateArray.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
        allocateArray.NumChannels = 1;
        cuArray3DCreate(d_volumeArray, allocateArray);

        // Copy the volume data data to the 3D array
        final CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
        copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copy.srcHost = Pointer.to(h_volume);
        copy.srcPitch = volumeSize.x;
        copy.srcHeight = volumeSize.y;
        copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copy.dstArray = d_volumeArray;
        copy.dstPitch = volumeSize.x;
        copy.dstHeight = volumeSize.y;
        copy.WidthInBytes = volumeSize.x;
        copy.Height = volumeSize.y;
        copy.Depth = volumeSize.z;
        cuMemcpy3D(copy);

        // Obtain the 3D texture reference for the volume data from
        // the module, set its parameters and assign the 3D volume
        // data array as its reference.
        cuModuleGetTexRef(tex, module, "tex");
        cuTexRefSetFilterMode(tex,
                CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(tex, 0,
                CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(tex, 1,
                CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFormat(tex,
                CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, 1);
        cuTexRefSetFlags(tex,
                CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetArray(tex, d_volumeArray,
                CU_TRSA_OVERRIDE_FORMAT);

        // The RGBA components of the transfer function texture
        final float transferFunc[] = new float[]
                {
                        0.0f, 0.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 1.0f,
                        1.0f, 0.5f, 0.0f, 1.0f,
                        1.0f, 1.0f, 0.0f, 1.0f,
                        0.0f, 1.0f, 0.0f, 1.0f,
                        0.0f, 1.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        1.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 0.0f
                };

        // Create the 2D (float4) array that will contain the
        // transfer function data.
        final CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
        ad.Width = transferFunc.length / 4;
        ad.Height = 1;
        ad.NumChannels = 4;
        cuArrayCreate(d_transferFuncArray, ad);

        // Copy the transfer function data to the array
        final CUDA_MEMCPY2D copy2 = new CUDA_MEMCPY2D();
        copy2.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copy2.srcHost = Pointer.to(transferFunc);
        copy2.srcPitch = transferFunc.length * Sizeof.FLOAT;
        copy2.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copy2.dstArray = d_transferFuncArray;
        copy2.WidthInBytes = transferFunc.length * Sizeof.FLOAT;
        copy2.Height = 1;
        cuMemcpy2D(copy2);

        // Obtain the transfer texture reference from the module,
        // set its parameters and assign the transfer function
        // array as its reference.
        cuModuleGetTexRef(transferTex, module, "transferTex");
        cuTexRefSetFilterMode(transferTex,
                CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(transferTex, 0,
                CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(transferTex,
                CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(transferTex,
                CUarray_format.CU_AD_FORMAT_FLOAT, 4);
        cuTexRefSetArray(transferTex, d_transferFuncArray,
                CU_TRSA_OVERRIDE_FORMAT);

        // Set the texture references as parameters for the function call
        cuParamSetTexRef(function, CU_PARAM_TR_DEFAULT,
                tex);
        cuParamSetTexRef(function, CU_PARAM_TR_DEFAULT,
                transferTex);
    }

    /**
     * Creates a pixel buffer object (PBO) which stores the image that
     * is created by the kernel, and which will later be rendered
     * by JOGL.
     *
     * @param gl The GL context
     */
    private void initPBO(final GL gl) {
        if (pbo != 0) {
            cuGLUnregisterBufferObject(pbo);
            gl.glDeleteBuffers(1, new int[]{pbo}, 0);
            pbo = 0;
        }

        // Create and bind a pixel buffer object with the current
        // width and height of the rendering component.
        final int pboArray[] = new int[1];
        gl.glGenBuffers(1, pboArray, 0);
        pbo = pboArray[0];
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, pbo);
        gl.glBufferData(GL2.GL_PIXEL_UNPACK_BUFFER,
                width * height * Sizeof.BYTE * 4, null, GL.GL_DYNAMIC_DRAW);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

        // Register the PBO for usage with CUDA
        cuGLRegisterBufferObject(pbo);

        // Calculate new grid size
        gridSize = new dim3(
                iDivUp(width, blockSize.x),
                iDivUp(height, blockSize.y), 1);
    }

    /**
     * Integral division, rounding the result to the next highest integer.
     *
     * @param a Dividend
     * @param b Divisor
     * @return a/b rounded to the next highest integer.
     */
    private static int iDivUp(final int a, final int b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    /**
     * Set up a default view for the given GLAutoDrawable
     *
     * @param drawable The GLAutoDrawable to set the view for
     */
    private void setupView(final GLAutoDrawable drawable) {
        final GL2 gl = drawable.getGL().getGL2();

        gl.glViewport(0, 0, drawable.getWidth(), drawable.getHeight());

        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glLoadIdentity();

        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glLoadIdentity();
        gl.glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }

    /**
     * Call the kernel function, rendering the 3D volume data image
     * into the PBO
     */
    private void render() {
        // Map the PBO to get a CUDA device pointer
        final CUdeviceptr d_output = new CUdeviceptr();
        cuGLMapBufferObject(d_output, new long[1], pbo);
        cuMemsetD32(d_output, 0, width * height);

        // Set up the execution parameters for the kernel:
        // - One pointer for the output that is mapped to the PBO
        // - Two ints for the width and height of the image to render
        // - Four floats for the visualization parameters of the renderer
        final Pointer dOut = Pointer.to(d_output);
        final Pointer pWidth = Pointer.to(new int[]{width});
        final Pointer pHeight = Pointer.to(new int[]{height});
        final Pointer pDensity = Pointer.to(new float[]{density});
        final Pointer pBrightness = Pointer.to(new float[]{brightness});
        final Pointer pTransferOffset = Pointer.to(new float[]{transferOffset});
        final Pointer pTransferScale = Pointer.to(new float[]{transferScale});

        int offset = 0;

        offset = align(offset, Sizeof.POINTER);
        cuParamSetv(function, offset, dOut, Sizeof.POINTER);
        offset += Sizeof.POINTER;

        offset = align(offset, Sizeof.INT);
        cuParamSetv(function, offset, pWidth, Sizeof.INT);
        offset += Sizeof.INT;

        offset = align(offset, Sizeof.INT);
        cuParamSetv(function, offset, pHeight, Sizeof.INT);
        offset += Sizeof.INT;

        offset = align(offset, Sizeof.FLOAT);
        cuParamSetv(function, offset, pDensity, Sizeof.FLOAT);
        offset += Sizeof.FLOAT;

        offset = align(offset, Sizeof.FLOAT);
        cuParamSetv(function, offset, pBrightness, Sizeof.FLOAT);
        offset += Sizeof.FLOAT;

        offset = align(offset, Sizeof.FLOAT);
        cuParamSetv(function, offset, pTransferOffset, Sizeof.FLOAT);
        offset += Sizeof.FLOAT;

        offset = align(offset, Sizeof.FLOAT);
        cuParamSetv(function, offset, pTransferScale, Sizeof.FLOAT);
        offset += Sizeof.FLOAT;

        cuParamSetSize(function, offset);

        // Call the CUDA kernel, writing the results into the PBO
        cuFuncSetBlockShape(function, blockSize.x, blockSize.y, 1);
        cuLaunchGrid(function, gridSize.x, gridSize.y);
        cuCtxSynchronize();
        cuGLUnmapBufferObject(pbo);
    }

    /**
     * Implementation of GLEventListener: Called when the given GLAutoDrawable
     * is to be displayed.
     */
    @Override
    public void display(final GLAutoDrawable drawable) {
        final GL2 gl = drawable.getGL().getGL2();

        // Use OpenGL to build view matrix
        final float modelView[] = new float[16];
        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
        gl.glRotatef(-rotationY, 0.0f, 1.0f, 0.0f);
        gl.glTranslatef(-translationX, -translationY, -translationZ);
        gl.glGetFloatv(GL2.GL_MODELVIEW_MATRIX, modelView, 0);
        gl.glPopMatrix();

        // Build the inverted view matrix
        invViewMatrix[0] = modelView[0];
        invViewMatrix[1] = modelView[4];
        invViewMatrix[2] = modelView[8];
        invViewMatrix[3] = modelView[12];
        invViewMatrix[4] = modelView[1];
        invViewMatrix[5] = modelView[5];
        invViewMatrix[6] = modelView[9];
        invViewMatrix[7] = modelView[13];
        invViewMatrix[8] = modelView[2];
        invViewMatrix[9] = modelView[6];
        invViewMatrix[10] = modelView[10];
        invViewMatrix[11] = modelView[14];

        // Copy the inverted view matrix to the global variable that
        // was obtained from the module. The inverted view matrix
        // will be used by the kernel during rendering.
        cuMemcpyHtoD(c_invViewMatrix, Pointer.to(invViewMatrix),
                invViewMatrix.length * Sizeof.FLOAT);

        // Render and fill the PBO with pixel data
        render();

        // Draw the image from the PBO
        gl.glClear(GL.GL_COLOR_BUFFER_BIT);
        gl.glDisable(GL.GL_DEPTH_TEST);
        gl.glRasterPos2i(0, 0);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, pbo);
        gl.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 0);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

        // Update FPS information in main frame title
        step++;
        final long currentTime = System.nanoTime();
        if (prevTimeNS == -1) {
            prevTimeNS = currentTime;
        }
        final long diff = currentTime - prevTimeNS;
        if (diff > 1e9) {
            final double fps = (diff / 1e9) * step;
            String t = "JCuda 3D texture volume rendering sample - ";
            t += String.format("%.2f", fps) + " FPS";
            frame.setTitle(t);
            prevTimeNS = currentTime;
            step = 0;
        }

    }

    /**
     * Implementation of GLEventListener: Called then the GLAutoDrawable was
     * reshaped
     */
    @Override
    public void reshape(
            final GLAutoDrawable drawable, final int x, final int y, final int width, final int height) {
        this.width = width;
        this.height = height;

        initPBO(drawable.getGL());

        setupView(drawable);
    }

    /**
     * Implementation of GLEventListener - not used
     */
    @Override
    public void dispose(final GLAutoDrawable arg0) {
    }

    /**
     * Stops the animator and calls System.exit() in a new Thread.
     * (System.exit() may not be called synchronously inside one
     * of the JOGL callbacks)
     */
    private void runExit() {
        new Thread(new Runnable() {
            public void run() {
                animator.stop();
                System.exit(0);
            }
        }).start();
    }

}