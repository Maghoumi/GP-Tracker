package utils.cuda.datatypes;

import java.awt.Color;

/**
 * Represents an CUDA float4 struct
 * Can also represent an RGBA color object
 * 
 * @author Mehran Maghoumi
 */
public class Float4 {
	
	/**
	 * The x component of a float4 struct
	 * (Equivalent to the red component of the ARGB color model)
	 */
	public float x;
	
	/**
	 * The y component of a float4 struct
	 * (Equivalent to the green component of the ARGB color model)
	 */
	public float y;
	
	/**
	 * The z component of a float4 struct
	 * (Equivalent to the blue component of the ARGB color model)
	 */
	public float z;
	
	/**
	 * The w component of a float4 struct
	 * (Equivalent to the alpha component of the ARGB color model)
	 */
	public float w;
	
	public Float4(float x, float y, float z, float w) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}
	
	public Float4(float x, float y, float z) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = 255;
	}
	
	public Float4(int x, int y, int z, int w) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}
	
	/**
	 * Create an instance of this class by parsing the ARGB value
	 * 
	 * @param rgb
	 */
	public Float4(int rgb) {
		this.x = rgb >> 16 & 0xff;
		this.y = rgb>> 8 & 0xff;
		this.z = rgb & 0xff;
		this.w = rgb >> 24 & 0xff;
	}
	
	/**
	 * Copy constructor
	 * @param in
	 */
	public Float4(Float4 in) {
		this.x = in.x;
		this.y = in.y;
		this.z = in.z;
		this.w = in.w;
	}
	
	/**
	 * Converts this instance of this object to an instance of a color object
	 * @return
	 */
	public Color toColor() {
		int red = (int) this.x;
		int green = (int) this.y;
		int blue = (int) this.z;
		int alpha = (int) this.w;
		
		red = red > 255 ? 255 : red;
		green = green > 255 ? 255 : green;
		blue = blue > 255 ? 255 : blue;
		alpha = alpha > 255 ? 255 : alpha;
		
		red = red < 0 ? 0 : red;
		green = green < 0 ? 0 : green;
		blue = blue < 0 ? 0 : blue;
		alpha = alpha < 0 ? 0 : alpha;
		
		return new Color(red, green, blue, 255);
	}	 
	
	
	/**
	 * Returns the red value of this instance of the object
	 * @return
	 */
	public float getRed() {
		return this.x;
	}
	
	/**
	 * Returns the green value of this instance of the object
	 * @return
	 */
	public float getGreen() {
		return this.y;
	}
	
	/**
	 * Returns the blue value of this instance of the object
	 * @return
	 */
	public float getBlue() {
		return this.z;
	}
	
	/**
	 * Returns the alpha value of this instance of the object
	 * @return
	 */
	public float getAlpha() {
		return this.w;
	}
	
	/**
	 * Sets the red color channel for this instance of the object
	 * @param red
	 */
	public void setRed(float red) {
		this.x = red;
	}
	
	/**
	 * Sets the green color channel for this instance of the object
	 * @param green
	 */
	public void setGreen(float green) {
		this.y = green;
	}
	
	/**
	 * Sets the blue color channel for this instance of the object
	 * @param blue
	 */
	public void setblue(float blue) {
		this.z = blue;
	}
	
	/**
	 * Sets the alpha color channel for this instance of the object
	 * @param alpha
	 */
	public void setAlpha(float alpha) {
		this.w = alpha;
	}
	
	public Float4 clone() {
		return new Float4(this);
	}
	
	/**
	 * Gets the color of the selected pixel at the specified linear location
	 * and returns it as a Float4 object. NOTE: Does not change channel definitions!
	 * Previously, it was convoluting the channels, but no MORE OF THIS CRAP!
	 * 
	 * @param position
	 * 		Linear position of the pixel
	 * @return
	 * 		Color of the specified pixel, as a Float4 object
	 */
	public static Float4 getFloat4(float[] input, int position) {
		float x = 0, y = 0, z = 0, w = 0;
		try {
		int index = position * 4; // Because each pixel takes 4 banks in the byte array
		x = input[index];
		y = input[index + 1];
		z = input[index + 2];
		w = input[index + 3];
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return new Float4(x, y, z, w);
	}
	
	/**
	 * Gets a float array and an input Float4 instance and fills the RGB values in
	 * the float array. Useful for transferring stuff to CUDA.
	 * Returns the new index for the array to use. (I know, I've written shit)
	 * 
	 * @param result
	 * @param input
	 * @param index
	 */
	public static int fillArray3ChannelABGR(float[] result, Float4 input, int index) {
		// ABGR => BGR ==> x, y, z, w ==> y, z, w
		result[index++] = input.y; // 
		result[index++] = input.z;
		result[index++] = input.w;
		
		return index;
	}
	
	public static float[] float4ABGRToFloat3flat(float[] input) {
		int size = (int) (input.length - (0.25) * input.length);
		float[] result = new float[size];
		
		int j = 0;
		for (int i = 0 ; i < input.length ; i++)
			if (i % 4 != 0)
				result[j++] = input[i];
		
		return result;
	}
}