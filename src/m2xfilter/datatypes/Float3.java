package m2xfilter.datatypes;

import java.awt.Color;

/**
 * Represents an CUDA float3 struct
 * Can also represent RED-GREEN-BLUE color space
 * Note: the order is totally dependent on you!
 * 
 * @author Mehran Maghoumi
 */
public class Float3{
	
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
	
	public Float3(float x, float y, float z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	public Float3(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	/**
	 * Copy constructor
	 * @param in
	 */
	public Float3(Float3 in) {
		this.x = in.x;
		this.y = in.y;
		this.z = in.z;
	} 
	
	/**
	 * Converts this instance of this object to an instance of a color object
	 * @return
	 */
	public Color toColor() {
		int red = (int) this.x;
		int green = (int) this.y;
		int blue = (int) this.z;
		
		red = red > 255 ? 255 : red;
		green = green > 255 ? 255 : green;
		blue = blue > 255 ? 255 : blue;
		
		red = red < 0 ? 0 : red;
		green = green < 0 ? 0 : green;
		blue = blue < 0 ? 0 : blue;
		
		return new Color(red, green, blue);
	}	 
	
	public Float3 clone() {
		return new Float3(this);
	}
	
	/**
	 * Gets the color of the selected pixel at the specified linear location
	 * and returns it as a Float3 object. NOTE: Does not change channel definitions!
	 * Previously, it was convoluting the channels, but no MORE OF THIS CRAP!
	 * 
	 * @param position
	 * 		Linear position of the pixel
	 * @return
	 * 		Color of the specified pixel, as a Float4 object
	 */
	public static Float3 getFloat3(float[] input, int position) {
		int index = position * 3; // Because each pixel takes 3 banks in the byte array
		float x = input[index];
		float y = input[index + 1];
		float z = input[index + 2];
		
		return new Float3(x, y, z);
	}
}