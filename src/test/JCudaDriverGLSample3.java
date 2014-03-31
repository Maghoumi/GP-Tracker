package test;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */

import static jcuda.driver.CUgraphicsMapResourceFlags.*;
import static jcuda.driver.JCudaDriver.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.*;
import java.util.Arrays;

import javax.media.opengl.*;
import javax.media.opengl.awt.GLCanvas;
import javax.swing.*;

import jcuda.*;
import jcuda.driver.*;

import com.jogamp.opengl.util.Animator;

/**
 * This class demonstrates how to use the JCudaDriver GL bindings API 
 * to interact with JOGL, the Java Bindings for OpenGL. It creates
 * a vertex buffer object (VBO) consisting of a rectangular grid of
 * points, and animates it with a sine wave. Is intended to be used 
 * with JOGL 2, and uses only the OpenGL 3.2 core profile and GLSL 1.5. 
 * <br />
 * Pressing the 't' key will toggle between the CUDA computation and
 * the Java computation mode.
 * <br />
 * This sample actually uses the kernel that is created for the 
 * "Simple OpenGL" sample from the NVIDIA CUDA code samples web site.
 */
public class JCudaDriverGLSample3 implements GLEventListener
{
    /**
     * Entry point for this sample.
     * 
     * @param args not used
     */
    public static void main(String args[])
    {
        GLProfile profile = GLProfile.get(GLProfile.GL3);
        final GLCapabilities capabilities = new GLCapabilities(profile);
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                new JCudaDriverGLSample3(capabilities);
            }
        });
    }

    /**
     * The source code for the vertex shader
     */
    private static String vertexShaderSource = 
        "#version 150 core" + "\n" +
        "in  vec4 inVertex;" + "\n" +
        "in  vec3 inColor;" + "\n" +
        "uniform mat4 modelviewMatrix;" + "\n" +
        "uniform mat4 projectionMatrix;" + "\n" +
        "void main(void)" + "\n" +
        "{" + "\n" +
        "    gl_Position = " + "\n" +
        "        projectionMatrix * modelviewMatrix * inVertex;" + "\n" +
        "}";
    
    /**
     * The source code for the fragment shader
     */
    private static String fragmentShaderSource =
        "#version 150 core" + "\n" +
        "out vec4 outColor;" + "\n" +
        "void main(void)" + "\n" +
        "{" + "\n" +
        "    outColor = vec4(1.0,0.0,0.0,1.0);" + "\n" +
        "}";
    
    /**
     * The width segments of the mesh to be displayed.
     * Should be a multiple of 8.
     */
    private static final int meshWidth = 8 * 64;

    /**
     * The height segments of the mesh to be displayed
     * Should be a multiple of 8.
     */
    private static final int meshHeight = 8 * 64;

    /**
     * The VAO identifier
     */
    private int vertexArrayObject;
    
    /**
     * The VBO identifier
     */
    private int vertexBufferObject;

    /**
     * The Graphics resource associated with the VBO 
     */
    private CUgraphicsResource vboGraphicsResource;
    
    /**
     * The current animation state of the mesh
     */
    private float animationState = 0.0f;

    /**
     * The animator used to animate the mesh.
     */
    private Animator animator;

    /**
     * The handle for the CUDA function of the kernel that is to be called
     */
    private CUfunction function;

    /**
     * Whether the computation should be performed with CUDA or
     * with Java. May be toggled by pressing the 't' key.
     */
    private boolean useCUDA = true;

    /**
     * The ID of the OpenGL shader program
     */
    private int shaderProgramID;
    
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
    private float rotationX = 40;

    /**
     * The rotation about the Y-axis, in degrees
     */
    private float rotationY = 30;

    /**
     * The current projection matrix
     */
    float projectionMatrix[] = new float[16];

    /**
     * The current projection matrix
     */
    float modelviewMatrix[] = new float[16];
    
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
    class MouseControl implements MouseMotionListener, MouseWheelListener
    {
        private Point previousMousePosition = new Point();

        @Override
        public void mouseDragged(MouseEvent e)
        {
            int dx = e.getX() - previousMousePosition.x;
            int dy = e.getY() - previousMousePosition.y;

            // If the left button is held down, move the object
            if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) == 
                MouseEvent.BUTTON1_DOWN_MASK)
            {
                translationX += dx / 100.0f;
                translationY -= dy / 100.0f;
            }

            // If the right button is held down, rotate the object
            else if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) == 
                MouseEvent.BUTTON3_DOWN_MASK)
            {
                rotationX += dy;
                rotationY += dx;
            }
            previousMousePosition = e.getPoint();
            updateModelviewMatrix();
        }

        @Override
        public void mouseMoved(MouseEvent e)
        {
            previousMousePosition = e.getPoint();
        }

        @Override
        public void mouseWheelMoved(MouseWheelEvent e)
        {
            // Translate along the Z-axis
            translationZ += e.getWheelRotation() * 0.25f;
            previousMousePosition = e.getPoint();
            updateModelviewMatrix();
        }
    }

    
    /**
     * Inner class extending a KeyAdapter for the keyboard
     * interaction
     */
    class KeyboardControl extends KeyAdapter
    {
        public void keyTyped(KeyEvent e)
        {
            char c = e.getKeyChar();
            if (c == 't')
            {
                useCUDA = !useCUDA;
            }
        }
    }

    /**
     * Creates a new JCudaDriverGLSample3.
     */
    public JCudaDriverGLSample3(GLCapabilities capabilities)
    {
        // Initialize the GL component and the animator
        GLCanvas glComponent = new GLCanvas(capabilities);
        glComponent.setFocusable(true);
        glComponent.addGLEventListener(this);

        // Initialize the mouse and keyboard controls
        MouseControl mouseControl = new MouseControl();
        glComponent.addMouseMotionListener(mouseControl);
        glComponent.addMouseWheelListener(mouseControl);
        KeyboardControl keyboardControl = new KeyboardControl();
        glComponent.addKeyListener(keyboardControl);
        updateModelviewMatrix();

        // Create the main frame 
        frame = new JFrame("JCuda / JOGL interaction sample");
        frame.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                runExit();
            }
        });
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(800, 800));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
        glComponent.requestFocus();

        // Create and start the animator
        animator = new Animator(glComponent);
        animator.start();
        
    }

    /**
     * Update the modelview matrix depending on the
     * current translation and rotation
     */
    private void updateModelviewMatrix()
    {
        float m0[] = translation(translationX, translationY, translationZ);
        float m1[] = rotationX(rotationX);
        float m2[] = rotationY(rotationY);
        modelviewMatrix = multiply(multiply(m1,m2), m0);
    }
    
    /**
     * Implementation of GLEventListener: Called to initialize the 
     * GLAutoDrawable
     */
    @Override
    public void init(GLAutoDrawable drawable)
    {
        // Perform the default GL initialization 
        GL3 gl = drawable.getGL().getGL3();
        gl.setSwapInterval(0);
        gl.glEnable(GL3.GL_DEPTH_TEST);
        gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        setupView(drawable);

        // Initialize the shaders
        initShaders(gl);

        // Initialize JCuda
        initJCuda();

        // Create the VBO containing the vertex data
        initVBO(gl);
    }

    
    /**
     * Initialize the shaders and the shader program
     * 
     * @param gl The GL context
     */
    private void initShaders(GL3 gl)
    {
        shaderProgramID = gl.glCreateProgram();

        int vertexShaderID = gl.glCreateShader(GL3.GL_VERTEX_SHADER);
        gl.glShaderSource(vertexShaderID, 1, 
            new String[]{vertexShaderSource}, null);
        gl.glCompileShader(vertexShaderID);
        gl.glAttachShader(shaderProgramID, vertexShaderID);
        gl.glDeleteShader(vertexShaderID);

        int fragmentShaderID = gl.glCreateShader(GL3.GL_FRAGMENT_SHADER);
        gl.glShaderSource(fragmentShaderID, 1, 
            new String[]{fragmentShaderSource}, null);
        gl.glCompileShader(fragmentShaderID);
        gl.glAttachShader(shaderProgramID, fragmentShaderID);
        gl.glDeleteShader(fragmentShaderID);
        
        gl.glLinkProgram(shaderProgramID);
        gl.glValidateProgram(shaderProgramID);
    }
    
    /**
     * Initialize the JCudaDriver. Note that this has to be done from the
     * same thread that will later use the JCudaDriver API
     */
    private void initJCuda()
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Create a device and a context
        cuInit(0);
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        CUcontext glCtx = new CUcontext();
        cuCtxCreate(glCtx, 0, dev);

        // Prepare the PTX file containing the kernel
        String ptxFileName = "";
        try
        {
            ptxFileName = preparePtxFile("bin/test/simpleGL_kernel.cu");
        }
        catch (IOException e)
        {
            System.err.println("Could not create PTX file");
            throw new RuntimeException("Could not create PTX file", e);
        }
        
        // Load the PTX file containing the kernel
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the kernel function. This function
        // will later be called during the animation, in the display 
        // method of this GLEventListener.
        function = new CUfunction();
        cuModuleGetFunction(function, module,
            "_Z6kernelP6float4jjf");
    }
    

    
    /**
     * Create the vertex buffer object (VBO) that stores the
     * vertex positions.
     * 
     * @param gl The GL context
     */
    private void initVBO(GL3 gl)
    {
        int buffer[] = new int[1];

        // Create the vertex buffer object
        gl.glGenVertexArrays(1, IntBuffer.wrap(buffer));
        vertexArrayObject = buffer[0];

        gl.glBindVertexArray(vertexArrayObject);
        
        // Create the vertex buffer object
        gl.glGenBuffers(1, IntBuffer.wrap(buffer));
        vertexBufferObject = buffer[0];

        // Initialize the vertex buffer object
        gl.glBindBuffer(GL.GL_ARRAY_BUFFER, vertexBufferObject);
        int size = meshWidth * meshHeight * 4 * Sizeof.FLOAT;
        gl.glBufferData(GL.GL_ARRAY_BUFFER, size, (Buffer) null,
            GL.GL_DYNAMIC_DRAW);

        // Initialize the attribute location of the input
        // vertices for the shader program
        int location = gl.glGetAttribLocation(shaderProgramID, "inVertex");
        gl.glVertexAttribPointer(location, 4, GL3.GL_FLOAT, false, 0, 0);
        gl.glEnableVertexAttribArray(location);

        // Register the vertexBufferObject for use with CUDA
        vboGraphicsResource = new CUgraphicsResource();
        cuGraphicsGLRegisterBuffer(
            vboGraphicsResource, vertexBufferObject, 
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
    }
    

    /**
     * Set up a default view for the given GLAutoDrawable
     * 
     * @param drawable The GLAutoDrawable to set the view for
     */
    private void setupView(GLAutoDrawable drawable)
    {
        GL3 gl = drawable.getGL().getGL3();
        gl.glViewport(0, 0, drawable.getWidth(), drawable.getHeight());
        float aspect = (float) drawable.getWidth() / drawable.getHeight();
        projectionMatrix = perspective(50, aspect, 0.1f, 100.0f);
    }

    /**
     * Implementation of GLEventListener: Called when the given GLAutoDrawable
     * is to be displayed.
     */
    @Override
    public void display(GLAutoDrawable drawable)
    {
        GL3 gl = drawable.getGL().getGL3();

        if (useCUDA)
        {
            // Run the CUDA kernel to generate new vertex positions.
            runCuda(gl);
        }
        else
        {
            // Run the Java method to generate new vertex positions.
            runJava(gl);
        }

        gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);

        // Activate the shader program
        gl.glUseProgram(shaderProgramID);
        
        // Set the current projection matrix
        int projectionMatrixLocation = 
            gl.glGetUniformLocation(shaderProgramID, "projectionMatrix");
        gl.glUniformMatrix4fv(
            projectionMatrixLocation, 1, false, projectionMatrix, 0);

        // Set the current modelview matrix
        int modelviewMatrixLocation = 
            gl.glGetUniformLocation(shaderProgramID, "modelviewMatrix");
        gl.glUniformMatrix4fv(
            modelviewMatrixLocation, 1, false, modelviewMatrix, 0);
        
        // Render the VBO
        gl.glBindBuffer(GL3.GL_ARRAY_BUFFER, vertexBufferObject);
        gl.glDrawArrays(GL3.GL_POINTS, 0, meshWidth * meshHeight);
        
        // Update FPS information in main frame title
        step++;
        long currentTime = System.nanoTime();
        if (prevTimeNS == -1)
        {
            prevTimeNS = currentTime;
        }
        long diff = currentTime - prevTimeNS;
        if (diff > 1e9)
        {
            double fps = (diff / 1e9) * step;
            String t = "JCuda / JOGL interaction sample - ";
            t += useCUDA?"JCuda":"Java";
            t += " mode: "+String.format("%.2f", fps)+" FPS";
            frame.setTitle(t);
            prevTimeNS = currentTime;
            step = 0;
        }

        animationState += 0.01;
    }

    /**
     * Run the CUDA computation to create new vertex positions
     * inside the vertexBufferObject.
     * 
     * @param gl The current GL.
     */
    private void runCuda(GL gl)
    {
        // Map the vertexBufferObject for writing from CUDA.
        // The basePointer will afterwards point to the
        // beginning of the memory area of the VBO.
        CUdeviceptr basePointer = new CUdeviceptr();
        cuGraphicsMapResources(
            1, new CUgraphicsResource[]{vboGraphicsResource}, null);
        cuGraphicsResourceGetMappedPointer(
            basePointer, new long[1], vboGraphicsResource);
        
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values. One 
        // pointer to the base pointer of the geometry data, 
        // one int for the mesh width, one int for the mesh 
        // height, and one float for the current animation state. 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(basePointer),
            Pointer.to(new int[]{meshWidth}),
            Pointer.to(new int[]{meshHeight}),
            Pointer.to(new float[]{animationState})
        );

        // Call the kernel function.
        int blockX = 8;
        int blockY = 8;
        int gridX = meshWidth / blockX;
        int gridY = meshHeight / blockY;
        cuLaunchKernel(function,
            gridX, gridY, 1,       // Grid dimension
            blockX, blockY, 1,     // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        // Unmap buffer object
        cuGraphicsUnmapResources(
            1, new CUgraphicsResource[]{vboGraphicsResource}, null);
    }

    
    /**
     * Run the Java computation to create new vertex positions
     * inside the vertexBufferObject.
     * 
     * @param gl The current GL.
     */
    private void runJava(GL gl)
    {
        gl.glBindBuffer(GL.GL_ARRAY_BUFFER, vertexBufferObject);
        ByteBuffer byteBuffer = 
            gl.glMapBuffer(GL.GL_ARRAY_BUFFER, GL2.GL_READ_WRITE);
        if (byteBuffer == null)
        {
            throw new RuntimeException("Unable to map buffer");
        }
        FloatBuffer vertices = 
            byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (int x = 0; x < meshWidth; x++)
        {
            for (int y = 0; y < meshHeight; y++)
            {
                // Calculate u/v coordinates
                float u = x / (float) meshWidth;
                float v = y / (float) meshHeight;

                u = u * 2.0f - 1.0f;
                v = v * 2.0f - 1.0f;

                // Calculate simple sine wave pattern
                float freq = 4.0f;
                float w = (float) Math.sin(u * freq + animationState) *
                          (float) Math.cos(v * freq + animationState) * 0.5f;

                // Write output vertex
                int index = 4 * (y * meshWidth + x);
                vertices.put(index + 0, u);
                vertices.put(index + 1, w);
                vertices.put(index + 2, v);
                vertices.put(index + 3, 1);
            }
        }
        gl.glUnmapBuffer(GL.GL_ARRAY_BUFFER);
        gl.glBindBuffer(GL.GL_ARRAY_BUFFER, 0);
    }

    /**
     * Implementation of GLEventListener: Called then the 
     * GLAutoDrawable was reshaped
     */
    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width,
                    int height)
    {
        setupView(drawable);
    }

    /**
     * Implementation of GLEventListener - not used
     */
    @Override
    public void dispose(GLAutoDrawable drawable)
    {
    }

    /**
     * Stops the animator and calls System.exit() in a new Thread.
     * (System.exit() may not be called synchronously inside one
     * of the JOGL callbacks)
     */
    private void runExit()
    {
        new Thread(new Runnable()
        {
            @Override
            public void run()
            {
                animator.stop();
                System.exit(0);
            }
        }).start();
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
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
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
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    
    
    //=== Helper functions for matrix operations ==============================

    /**
     * Helper method that creates a perspective matrix
     * @param fovy The fov in y-direction, in degrees
     * 
     * @param aspect The aspect ratio
     * @param zNear The near clipping plane
     * @param zFar The far clipping plane
     * @return A perspective matrix
     */
    private static float[] perspective(
        float fovy, float aspect, float zNear, float zFar)
    {
        float radians = (float)Math.toRadians(fovy / 2);
        float deltaZ = zFar - zNear;
        float sine = (float)Math.sin(radians);
        if ((deltaZ == 0) || (sine == 0) || (aspect == 0)) 
        {
            return identity();
        }
        float cotangent = (float)Math.cos(radians) / sine;
        float m[] = identity();
        m[0*4+0] = cotangent / aspect;
        m[1*4+1] = cotangent;
        m[2*4+2] = -(zFar + zNear) / deltaZ;
        m[2*4+3] = -1;
        m[3*4+2] = -2 * zNear * zFar / deltaZ;
        m[3*4+3] = 0;
        return m;
    }
    
    /**
     * Creates an identity matrix
     * 
     * @return An identity matrix 
     */
    private static float[] identity()
    {
        float m[] = new float[16];
        Arrays.fill(m, 0);
        m[0] = m[5] = m[10] = m[15] = 1.0f;
        return m;
    }
    
    /**
     * Multiplies the given matrices and returns the result
     * 
     * @param m0 The first matrix
     * @param m1 The second matrix
     * @return The product m0*m1
     */
    private static float[] multiply(float m0[], float m1[])
    {
        float m[] = new float[16];
        for (int x=0; x < 4; x++)
        {
            for(int y=0; y < 4; y++)
            {
                m[x*4 + y] = 
                    m0[x*4+0] * m1[y+ 0] +
                    m0[x*4+1] * m1[y+ 4] +
                    m0[x*4+2] * m1[y+ 8] +
                    m0[x*4+3] * m1[y+12];
            }
        }
        return m;
    }
    
    /**
     * Creates a translation matrix
     * 
     * @param x The x translation
     * @param y The y translation
     * @param z The z translation
     * @return A translation matrix
     */
    private static float[] translation(float x, float y, float z)
    {
        float m[] = identity();
        m[12] = x;
        m[13] = y;
        m[14] = z;
        return m;
    }

    /**
     * Creates a matrix describing a rotation around the x-axis
     * 
     * @param angleDeg The rotation angle, in degrees
     * @return The rotation matrix
     */
    private static float[] rotationX(float angleDeg)
    {
        float m[] = identity();
        float angleRad = (float)Math.toRadians(angleDeg);
        float ca = (float)Math.cos(angleRad);
        float sa = (float)Math.sin(angleRad);
        m[ 5] =  ca;
        m[ 6] =  sa;
        m[ 9] = -sa;
        m[10] =  ca;
        return m;
    }

    /**
     * Creates a matrix describing a rotation around the y-axis
     * 
     * @param angleDeg The rotation angle, in degrees
     * @return The rotation matrix
     */
    private static float[] rotationY(float angleDeg)
    {
        float m[] = identity();
        float angleRad = (float)Math.toRadians(angleDeg);
        float ca = (float)Math.cos(angleRad);
        float sa = (float)Math.sin(angleRad);
        m[ 0] =  ca;
        m[ 2] = -sa;
        m[ 8] =  sa;
        m[10] =  ca;
        return m;
    }
    
}