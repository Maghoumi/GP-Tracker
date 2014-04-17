package utils.cuda.datatypes;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

import javax.imageio.ImageIO;

import jogamp.opengl.util.av.impl.FFMPEGMediaPlayer.PixelFormat;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

/**
 * Represents a 4 byte **ABGR** image in a byte[] array.
 * This class was meant to be compatible with CUDA input for image filters
 * 
 * @author Mehran Maghoumi
 *
 */
public class ByteImage implements Cloneable {
	
	public static final int NUM_CHANNELS = 4;

	/** The underlying BufferedImage that this class works on. */
	private BufferedImage underlyingImage;
	
	/** Provides easy access to the underlying image's canvas */
	private byte[] byteData;
	

	/**
	 * Initializes this instance using an input image. This constructor
	 * stores the width and the height of the original input so that the
	 * original image can be recreated using the byte[] array that keeps the
	 * underlying pixel values.
	 * 
	 * @param input
	 * 		The input image to be converted to the byte[] array
	 */
	public ByteImage(BufferedImage input) {
		// convert the image to a CUDA-compatible format
		this.underlyingImage = convert(input, BufferedImage.TYPE_4BYTE_ABGR);		
		DataBufferByte bufferBytes = (DataBufferByte) this.underlyingImage.getRaster().getDataBuffer();
		// Store the byte[] array of the converted image
		this.byteData = bufferBytes.getData();
	}
	
	/**
	 * Instantiates an instance of this class using OpenCV's IplImage object. Note that
	 * this only converts images from the BGR24 format
	 *  
	 * @param frame	An OpenCV IplImage
	 * @param pixelFormat	The pixel format of the source image
	 */
	public ByteImage(IplImage frame, int pixelFormat) {
		// Are we using BGR24?
		if (pixelFormat != PixelFormat.BGR24.ordinal())
			throw new RuntimeException("ByteImage only supports conversion from BGR24 frames!");
		
		ByteBuffer buffer = frame.getByteBuffer();
		int width = frame.width();
		int height = frame.height();
		
		this.underlyingImage = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR);
		DataBufferByte bufferBytes = (DataBufferByte) underlyingImage.getRaster().getDataBuffer();
		this.byteData = bufferBytes.getData();
		
		int dataIndex = 0;
		while (buffer.remaining() > 0) {
			this.byteData[dataIndex++] = (byte) 255;
			
			for (int i = 0 ; i < 3 ; i++)
				this.byteData[dataIndex++] = buffer.get();			
		}		
	}
	
	/**
	 * Creates an instance of this class using float[] data.
	 * Likely very inefficient
	 * 
	 * @param input
	 * @param width
	 * @param height
	 */
	public ByteImage(float[] input, int width, int height) {
		this.underlyingImage = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR);
		DataBufferByte bufferBytes = (DataBufferByte) underlyingImage.getRaster().getDataBuffer();
		this.byteData = bufferBytes.getData();
		
		for (int i = 0 ; i < input.length ; i++)
			byteData[i] = (byte)((int)input[i]);
	}
	
	/**
	 * Creates an instance of this class using byte[] data. Uses 
	 * System.arraycopy to copy the input elements to the backbone
	 * BufferedImage.
	 * 
	 * @param input
	 * @param width
	 * @param height
	 */
	public ByteImage(byte[] input, int width, int height) {
		this.underlyingImage = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR);
		DataBufferByte bufferBytes = (DataBufferByte) underlyingImage.getRaster().getDataBuffer();
		this.byteData = bufferBytes.getData();
		System.arraycopy(input,0,this.byteData,0,input.length);
	}
	
	public float[] getFloatData() {
		float[] result = new float[byteData.length];
		for (int i = 0 ; i < result.length ; i++)
			result[i] = byteData[i] & 0xFF;
		
		return result;
	}
	
	/**
	 * Returns the underlying BufferedImage of this instance of the class.
	 * Any update to the data on this instance of the class is always directly reflected 
	 * on the backbone BufferedImage. This method is efficient.
	 * 
	 * NOTE: Any change to the returned BufferedImage is also reflected on this instance
	 * of the class. In other words, the BufferedImage object is not a clone of the 
	 * underlying image that this class uses.
	 * 
	 * 
	 * @return	The BufferedImage version of this instance of the object
	 */
	public BufferedImage getBufferedImage() {
		return this.underlyingImage;
	}

	/**
	 * Returns a BufferedImage that has the specified type. If the given image
	 * has the specified type, it is returned. Otherwise, a new image with the
	 * specified type is created, filled with the input image, and returned.
	 * 
	 * @param inputImage
	 *            The input image
	 * @param targetType
	 *            The type
	 * @return The buffered image with the specified type
	 */
	private BufferedImage convert(BufferedImage inputImage, int targetType) {
		int w = inputImage.getWidth();
		int h = inputImage.getHeight();

		BufferedImage resultImage = new BufferedImage(w, h, targetType);
		Graphics2D g = resultImage.createGraphics();
		g.drawImage(inputImage, 0, 0, null);
		g.dispose();
		return resultImage;
	}
	
	// Accessor method for width
	public int getWidth() {
		return this.underlyingImage.getWidth();
	}
	
	// Accessor method for height
	public int getHeight() {
		return this.underlyingImage.getHeight();
	}
	
	/**
	 * @return	The size of the image in bytes (will return the
	 * 			underlying byte array's length)
	 */
	public int getSizeInBytes() {
		return this.byteData.length;
	}
	
	// Accessor method for the underlying byte data
	public byte[] getByteData() {
		return this.byteData;
	}
	
	public byte[] getByteDataWithoutAlpha() {
		// The final length is 3 quartes of the original length
		// because we are omitting one channel
		
		int length = (int) (0.75 * this.byteData.length);
		byte[] result = new byte[length];
		
		int j = 0;
		for (int i = 0 ; i < this.byteData.length ; i++)
			if (i % NUM_CHANNELS != 0)
				result[j++] = (byte) (this.byteData[i] & 0xFF);
		
		return result;
	}
	
	/**
	 * @return	The size of this image (i.e. width * height)
	 */
	public int getSize() {
		return this.getWidth() * this.getHeight();
	}
	
	/**
	 * Obtains a subimage of the original image and returns it as a new instance of 
	 * ByteImage. Note that unlike its BufferedImage equivalent method, the returned
	 * image does not share data buffers with the original ByteImage.
	 * 
	 * FIXME: This implementation is not very efficient especially since we are double-copying
	 * 
	 * @param x
	 * 		the X coordinate of the upper-left corner of the specified rectangular region
	 * @param y
	 * 		the Y coordinate of the upper-left corner of the specified rectangular region
	 * @param width
	 * 		the width of the specified rectangular region
	 * @param height
	 * 		the height of the specified rectangular region
	 * @return
	 * 	The subimage in the specified region.
	 */
	public ByteImage getSubimage(int x, int y , int width, int height) {
		BufferedImage subImg = this.underlyingImage.getSubimage(x, y, width, height);
		
		int srcPitch = this.underlyingImage.getWidth() * NUM_CHANNELS; // This is a 4 channel image
		int dstWidth = width;
		int dstHeight = height;
		int dstPitch = dstWidth * NUM_CHANNELS;
		
		// Allocate a new data
		byte[] subImgBuffer = new byte[dstPitch * dstHeight];
		
		int bufferCurrIndex = 0;
		for (int i = 0 ; i < dstHeight ; i++) {

			int yIndex = (y + i) * srcPitch + x * NUM_CHANNELS;
			System.arraycopy(this.byteData, yIndex, subImgBuffer, bufferCurrIndex, dstPitch);
			bufferCurrIndex += dstPitch;
		}
		
		return new ByteImage(subImgBuffer, subImg.getWidth(), subImg.getHeight());
	}
	
	/**
	 * Gets the color of the pixel at the specified coordinates
	 * 
	 * @param x
	 * 		x-coordinate of the pixel
	 * @param y
	 * 		y-coordinate of the pixel
	 * @return
	 * 		Color of the specified pixel
	 */
	public Color getColor(int x, int y) {
		return new Color(underlyingImage.getRGB(x, y));
	}
	
	/**
	 * Gets the color of the selected pixel at the specified linear location
	 * 
	 * @param position
	 * 		Linear position of the pixel
	 * @return
	 * 		Color of the specified pixel
	 */
	public Color getColor(int position) {
		int index = position * NUM_CHANNELS; // Because each pixel takes 4 banks in the byte array
		int a = this.byteData[index] & 0xFF;
		int b = this.byteData[index + 1] & 0xFF;
		int g = this.byteData[index + 2] & 0xFF;
		int r = this.byteData[index + 3] & 0xFF;
		
		return new Color(r, g, b, a);
	}
	
	/**
	 * Returns the number of channels for this instance of the object
	 * @return
	 */
	public int getNumChannels() {
		return NUM_CHANNELS;
	}
	
	/**
	 * Sets the color of the pixel at the specified coordinates
	 * 
	 * @param x
	 * 		x-coordinate of the pixel
	 * @param y
	 * 		y-coordinate of the pixel
	 */
	public void setColor(int x, int y, Color c) {
		this.underlyingImage.setRGB(x, y, c.getRGB());
	}
	
	/**
	 * Sets the color of the selected pixel at the specified linear location
	 * 
	 * @param position
	 * 		Linear position of the pixel
	 */
	public void setColor(int position, Color c) {
		int index = position * NUM_CHANNELS; // Because each pixel takes 4 banks in the byte array
		this.byteData[index] = (byte) c.getAlpha();
		this.byteData[index + 1] = (byte) c.getBlue();
		this.byteData[index + 2] = (byte) c.getGreen();
		this.byteData[index + 3] = (byte) c.getRed();
	}
	
	/**
	 * Gets the color of the selected pixel at the specified linear location
	 * 
	 * @param position
	 * 		Linear position of the pixel
	 * @return
	 * 		Color of the specified pixel
	 */
	public Float4 getFloat4(int position) {
		int index = position * NUM_CHANNELS; // Because each pixel takes 4 banks in the byte array
		int a = this.byteData[index] & 0xFF;
		int b = this.byteData[index + 1] & 0xFF;
		int g = this.byteData[index + 2] & 0xFF;
		int r = this.byteData[index + 3] & 0xFF;
		
		return new Float4(r, g, b, a);
	}
	
	/**
	 * Clones this instance by creating another BufferedImage from the data and then
	 * converting that to another instance of ByteImage.
	 */
	@Override
	public ByteImage clone() {
		return new ByteImage(getBufferedImage());
	}
	
	public static ByteImage loadFromFile(String path) throws IOException {
		return new ByteImage(ImageIO.read(new File(path)));
	}
	
	public static ByteImage loadFromFile(File f) throws IOException {
		return new ByteImage(ImageIO.read(f));
	}
}
