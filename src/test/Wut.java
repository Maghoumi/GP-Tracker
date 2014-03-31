package test;

import static jcuda.driver.JCudaDriver.*;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.IntBuffer;

import javax.imageio.ImageIO;
import javax.media.opengl.GL;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLDrawable;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.media.opengl.glu.GLU;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import utils.cuda.datatypes.ByteImage;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsMapResourceFlags;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

import com.jogamp.opengl.util.Animator;

public class Wut implements GLEventListener {
	
	private Frame frame;
	
	private CUdeviceptr devInput;
	private CUdeviceptr devOutput;
	
	private CUfunction function;
	
	int imageWidth;
	int imageHeight;
	
	
	int glBuffer;
	CUgraphicsResource bufferResource;
	int glTexture;
	
	public Wut(GLCapabilities capabilities) {
		// Initialize the GL component and the animator
        GLCanvas glComponent = new GLCanvas(capabilities);
        glComponent.setFocusable(true);
        glComponent.addGLEventListener(this);

        // Create the main frame 
        frame = new JFrame("WUT??");
        frame.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                //runExit();
            	System.out.println("Shutting down!");
            	System.exit(0);
            }
        });
        
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(800, 800));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
        glComponent.requestFocus();
	}

	@Override
	public void init(GLAutoDrawable drawable) {
		
		System.out.println("Init called");
		
		GL2 gl = drawable.getGL().getGL2(); 
        // set erase color
        gl.glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //white 
        // set drawing color and point size
//        gl.glColor3f(1.0f, 0.0f, 0.0f); 
        gl.glPointSize(4.0f); //a 'dot' is 4 by 4 pixels 		
        gl.glEnable(GL2.GL_DEPTH_TEST);
        
        initCuda();
        initBuffers(drawable);
        initTexture(drawable);
	}
	
	public void initCuda() {
		JCuda.setExceptionsEnabled(true);
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
            ptxFileName = preparePtxFile("bin/test/copy-cat.cu");
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
        cuModuleGetFunction(function, module, "copyme");
        
        BufferedImage image = null;
		try {
			image = ImageIO.read(new File("D:\\more-boats.png"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        ByteImage bi = new ByteImage(image);
        
        // ALlocate and transfer input image
        this.imageHeight = bi.getHeight();
        this.imageWidth = bi.getWidth();
        
        devInput = new CUdeviceptr();
        cuMemAlloc(devInput, imageWidth * imageHeight * 4 * Sizeof.BYTE);
        cuMemcpyHtoD(devInput, Pointer.to(bi.getByteData()), imageWidth * imageHeight * 4 * Sizeof.BYTE);
	}
	
	public void initBuffers(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		
		int[] buffer = new int[1];
		
		// Generate buffer
		gl.glGenBuffers(1, IntBuffer.wrap(buffer));
		glBuffer = buffer[0];
		
		// Bind the generated buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);
		// Specify the size of the buffer (no data is pre-loaded in this buffer)
		gl.glBufferData(GL2.GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, (Buffer)null, GL2.GL_DYNAMIC_DRAW);
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);
		
		bufferResource = new CUgraphicsResource();
		
		// Register buffer in CUDA
		cuGraphicsGLRegisterBuffer(bufferResource, glBuffer, CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
	}
	
	public void initTexture (GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		int[] texture = new int[1];
		
		gl.glEnable(GL2.GL_TEXTURE_2D);
		gl.glGenTextures(1, IntBuffer.wrap(texture));
		glTexture = texture[0];
		
		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);
		// FIXME RGBA8?!
		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_RGBA8, imageWidth, imageHeight, 0, GL2.GL_BGRA, GL2.GL_UNSIGNED_BYTE, (Buffer)null);
		
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_LINEAR);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_LINEAR);
		gl.glDisable(GL2.GL_TEXTURE_2D);
		
	}
	
	@Override
	public void display(GLAutoDrawable drawable) {
		System.out.println("Display called");
		
		runCuda(drawable);
		
		GL2 gl = drawable.getGL().getGL2();
		gl.glClear (GL.GL_COLOR_BUFFER_BIT);
		
//		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);
//		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);
//		gl.glTexSubImage2D(GL2.GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL2.GL_BGRA, GL2.GL_UNSIGNED_BYTE, 0);
//		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);
		
//		
//		
//		
		gl.glBegin (GL.GL_POINTS);
        	gl.glVertex2i (0,0);
        	gl.glVertex2f(0.5f, 0.5f);        	
        gl.glEnd ();		
		

//        gl.glBegin(GL2.GL_QUADS);
//        	gl.glTexCoord2f( 0, 1.0f); 
//	        gl.glVertex3f(0,0,0);
//	        
//	        gl.glTexCoord2f(0,0);
//	        gl.glVertex3f(0,1.0f,0);
//	        
//	        gl.glTexCoord2f(1.0f,0);
//	        gl.glVertex3f(1.0f,1.0f,0);
//	        
//	        gl.glTexCoord2f(1.0f,1.0f);
//	        gl.glVertex3f(1.0f,0,0);
//        gl.glEnd();
		
	}
	
	public void runCuda(GLAutoDrawable drawable) {
		devOutput = new CUdeviceptr();
		cuGraphicsMapResources(1, new CUgraphicsResource[]{bufferResource}, null);
        cuGraphicsResourceGetMappedPointer(devOutput, new long[1], bufferResource);
        
        Pointer kernelParams = Pointer.to(
                Pointer.to(devInput),
                Pointer.to(devOutput),
                Pointer.to(new int[]{imageWidth}),
                Pointer.to(new int[]{imageHeight})
            );
        
        final int BLOCK_SIZE = 16;
        
        cuLaunchKernel(function,
        		(imageWidth - 1) / BLOCK_SIZE + 1, (imageHeight - 1) / BLOCK_SIZE + 1, 1,
        		BLOCK_SIZE, BLOCK_SIZE, 1,
        		0, null,
        		kernelParams, null);
        
        cuCtxSynchronize();
        
        cuGraphicsUnmapResources(1, new CUgraphicsResource[]{bufferResource}, null);
	}

	@Override
	public void reshape(GLAutoDrawable drawable, int x, int y, int width,
			int height) {
		System.out.println("Reshape called");
		GL2 gl = drawable.getGL().getGL2();
		
		 
        gl.glViewport( 0, 0, width, height ); 
        gl.glMatrixMode( GL2.GL_PROJECTION );  
        gl.glLoadIdentity(); 
        gl.glOrtho(0,1.0f,0,1.0f,-1.0f,1.0f);
        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glLoadIdentity();
        
        //GLU glu = new GLU();
        //glu.gluOrtho2D( 0.0, width, height, 0.0); 
	}
	
	@Override
	public void dispose(GLAutoDrawable drawable) {/* TODO Auto-generated method stub*/}
	
	
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
//        if (ptxFile.exists())
//        {
//            return ptxFileName;
//        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -arch=sm_21 -ptx "+
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
	
	public static void main(String args[])
    {
        GLProfile profile = GLProfile.get(GLProfile.GL2);
        final GLCapabilities capabilities = new GLCapabilities(profile);
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                new Wut(capabilities);
            }
        });
    }

}
