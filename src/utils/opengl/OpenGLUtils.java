package utils.opengl;

import java.awt.Color;
import java.awt.Rectangle;
import java.nio.ByteBuffer;

import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;

import utils.ByteImage;

/**
 * Provides some utility functions for OpenGL in Java
 * 
 * @author Mehran Maghoumi
 *
 */
public class OpenGLUtils {
	
	/**
	 * Copies the content of a ByteImage to the OpenGL pixel buffer object
	 * @param frame	The image data to copy to the OpenGL buffer
	 * @param bufferObject	The handle to OpenGL's pixel buffer object
	 * @param drawable	The OpenGL drawable
	 */
	public static void copyBuffer(ByteImage frame, int bufferObject, GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		// Select buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, bufferObject);
		// Map buffer and obtain a ByteBuffer to the mapped buffer
		ByteBuffer byteBuffer = gl.glMapBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, GL2.GL_WRITE_ONLY);

		if (byteBuffer == null) {
			throw new RuntimeException("Unable to map buffer");
		}

		// Copy image to OpenGL buffer
		byteBuffer.put(frame.getByteData());

		// Unmap buffer
		gl.glUnmapBuffer(GL2.GL_PIXEL_UNPACK_BUFFER);
		// Unselect binded buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);
	}
	
	/**
	 * Draws a color overlay over the specified region of the specified pixel buffer object.
	 * 
	 * @param drawable	OpenGL drawbale
	 * @param bufferObject	The pixel buffer object to write to
	 * @param overlayColor	The overlay color
	 * @param opacity	The opacity of the overlay. Has to be a double in the range [0, 1], 1 meaning fully drawn overlay color
	 * @param imageWidth	The width of the frame (or image)
	 * @param imageHeight	The height of the frame (or image)
	 * @param roi	The region of interest. A rectangle that specifies the region to be overlayed
	 */
	public static void drawRegionOverlay(GLAutoDrawable drawable, int bufferObject,
				Color overlayColor, double opacity,
				int imageWidth, int imageHeight, Rectangle roi) {
		GL2 gl = drawable.getGL().getGL2();
		
		// Select buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, bufferObject);
		// Map buffer and obtain a ByteBuffer to the mapped buffer
		ByteBuffer byteBuffer = gl.glMapBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, GL2.GL_READ_WRITE);

		if (byteBuffer == null) {
			throw new RuntimeException("Unable to map buffer");
		}
		
		double colorMix = 1 - opacity; 
		Color overlayMix = new Color((int)(overlayColor.getRed() * opacity),
									(int)(overlayColor.getGreen() * opacity),
									(int)(overlayColor.getBlue() * opacity),
									255);

		// Colorize the specified buffer region
		for (int i = 0 ; i < roi.height ; i++) {

			int yIndex = (roi.y + i) * imageWidth * 4 + roi.x * 4;
			byteBuffer.position(yIndex);
			
			for (int j = 0 ; j < roi.width ; j++) {
				// Mark current position. Don't know why I have to do this and .position() won't work!
				byteBuffer.mark();
				
				// Grab the original color and overlay it using the given percentage
				int alpha = byteBuffer.get() & 0xFF; 
				int blue = (int) ((byteBuffer.get() & 0xFF) * colorMix + overlayMix.getBlue());
				int green = (int) ((byteBuffer.get() & 0xFF) * colorMix + overlayMix.getGreen());
				int red = (int) ((byteBuffer.get() & 0xFF)  * colorMix + overlayMix.getRed());
				Color finalColor = new Color(red, green, blue, alpha);
				
				// Reset to the marked position
				byteBuffer.reset();
				
				// Write the overlayed color
				byteBuffer.put((byte) (finalColor.getAlpha() & 0xFF));
				byteBuffer.put((byte) (finalColor.getBlue() & 0xFF));
				byteBuffer.put((byte) (finalColor.getGreen() & 0xFF));
				byteBuffer.put((byte) (finalColor.getRed() & 0xFF));
			}
		}
		
		// Unmap buffer
		gl.glUnmapBuffer(GL2.GL_PIXEL_UNPACK_BUFFER);
		// Unselect binded buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);
	}
	
	/**
	 * Draws the provided image on the specified region of the provided ByteBuffer. Useful for drawing images
	 * on an OpenGL pixel buffer object.
	 * 
	 * @param byteBuffer	The ByteBuffer to write to (usually, OpenGL's PBO)
	 * @param data	The image to draw
	 * @param bufferWidth	The width of the buffer
	 * @param bufferHeight	The height of the buffer
	 * @param roi	The region of interest (destination region)
	 */
	public static void drawOnBuffer(ByteBuffer byteBuffer, ByteImage data, int bufferWidth, int bufferHeight, Rectangle roi) {
		
		byte[] dataBytes = data.getByteData();
		int dataIndex = 0;
		int numChannels = data.getNumChannels();
		
		// Colorize the specified buffer region
		for (int i = 0 ; i < roi.height ; i++) {
	
			int yIndex = (roi.y + i) * bufferWidth * numChannels + roi.x * numChannels;
			byteBuffer.position(yIndex);
			
			for (int j = 0 ; j < roi.width * numChannels; j++) {
				byteBuffer.put(dataBytes[dataIndex++]);
			}
		}
	}
	
	/**
	 * Draws the provided image on the specified region of the provided byte[] buffer. Useful for drawing images
	 * on an OpenGL pixel buffer object. This method uses System.arraycopy() therefore the performance is expected
	 * to be better than the overload of drawOnBuffer which takes a ByteBuffer as an input.
	 * 
	 * @param byteBuffer	A byte[] buffer to write to (usually, OpenGL's PBO)
	 * @param data	The image to draw
	 * @param bufferWidth	The width of the buffer
	 * @param bufferHeight	The height of the buffer
	 * @param roi	The region of interest (destination region)
	 */
	public static void drawOnBuffer(byte[] byteBuffer, ByteImage data, int bufferWidth, int bufferHeight, Rectangle roi) {
		
		byte[] dataBytes = data.getByteData();
		int dataIndex = 0;
		int numChannels = data.getNumChannels();
		
		// Colorize the specified buffer region
		for (int i = 0 ; i < roi.height ; i++) {
	
			int yIndex = (roi.y + i) * bufferWidth * numChannels + roi.x * numChannels;
			
			System.arraycopy(dataBytes, dataIndex, byteBuffer, yIndex, roi.width * numChannels);
			dataIndex += roi.width * numChannels;
		}
	}
	
}
