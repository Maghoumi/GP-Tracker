package utils;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;

import javax.imageio.ImageIO;

public class ImageUtils {
	
	private static final double RED_WEIGHT = 0.2989;
	private static final double GREEN_WEIGHT = 0.5870;
	private static final double BLUE_WEIGHT = 0.1140;
	
	/**
	 * Returns a 2D array containing grayscale values of every pixel in the image
	 * 
	 * @param image
	 *            The input image
	 * @return 2D array containing RGB value of every pixel in the image
	 */
	public static int[][] toGrayscaleArray(BufferedImage image) {
		int width = image.getWidth(), height = image.getHeight();

		int[][] result = new int[height][width];
		
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Color c = new Color(image.getRGB(j, i));
				result[i][j] = (int) (RED_WEIGHT * c.getRed() + GREEN_WEIGHT * c.getGreen() + BLUE_WEIGHT * c.getBlue());
			}
		
		return result;
	}
	
	// I created it for ground truth problem
	public static int[][] toRGBArray(BufferedImage image) {
		int width = image.getWidth(), height = image.getHeight();

		int[][] result = new int[height][width];
		
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Color c = new Color(image.getRGB(j, i));
				result[i][j] = (int) (c.getRed() + c.getGreen() + c.getBlue());
			}
		
		return result;
	}

	/**
	 * Calculates the average value of RGB in the grid specified.
	 * 
	 * @param image
	 *            The input image
	 * @param x0
	 *            The x-coordinate of the center pixel
	 * @param y0
	 *            The x-coordinate of the center pixel
	 * @param x
	 *            The length of the grid
	 * @param y
	 *            The width of the grid
	 * @return The average RGB value for the grid
	 */
	public static Color meanRGB(BufferedImage image, int x0, int y0, int x, int y) {
		float red = 0, green = 0, blue = 0;
		
		int count = 0;

		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) {
				try {
					Color c = new Color(image.getRGB(x0 - (x / 2) + i, y0 - (y / 2) + j));
					
					red += c.getRed();
					green += c.getGreen();
					blue += c.getBlue();
					
					count++;
				} catch (Exception e) {
					// no perfect square grid was found around the pixel at (x0, y0)!
					// we will just ignore
				}
			}

		return new Color((int)red / count, (int)green / count, (int)blue / count);
	}

	/**
	 * Calculates the standard deviation of values of RGB in the grid specified.
	 * 
	 * @param image
	 *            The input image
	 * @param x0
	 *            The x-coordinate of the center pixel
	 * @param y0
	 *            The x-coordinate of the center pixel
	 * @param x
	 *            The length of the grid
	 * @param y
	 *            The width of the grid
	 * @return The standard deviation value for the grid
	 */
	public static Color standardDeviation(BufferedImage image, int x0, int y0, int x, int y) {
		Color averageColor = meanRGB(image, x0, y0, x, y);
		
		float red = 0, green = 0, blue = 0;
		int count = 0;

		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) {
				try {
					Color c = new Color(image.getRGB(x0 - (x / 2) + i, y0 - (y / 2) + j));
					red += Math.pow(c.getRed()- averageColor.getRed(), 2);
					green += Math.pow(c.getGreen()- averageColor.getGreen(), 2);
					blue += Math.pow(c.getBlue()- averageColor.getBlue(), 2);
					count++;
				} catch (Exception e) {
					// no perfect square grid was found around the pixel at (x0, y0)!
					// we will just ignore
				}
			}

		
		red = (float) Math.sqrt(red/count);
		green = (float) Math.sqrt(green/count);
		blue = (float) Math.sqrt(blue/count);
		
		return new Color((int)red, (int)green, (int)blue);
	}

	/**
	 * Gets the grayscale value of the pixel in the provided image at the
	 * provided coordinates
	 * 
	 * @param img
	 *            The input image
	 * @param x
	 *            x-coordinate
	 * @param y
	 *            y-coordinate
	 * @return
	 */
	public static int getTotalRGB(BufferedImage img, int x, int y) {
		Color c = new Color(img.getRGB(y, x));
		return (int) (RED_WEIGHT * c.getRed() + GREEN_WEIGHT * c.getGreen() + BLUE_WEIGHT * c.getBlue());
	}

	public static long countPixels(Color c, BufferedImage image) {
		long result = 0;
		System.out.println(image.getHeight() * image.getWidth());

		for (int i = 0; i < image.getWidth(); i++)
			for (int j = 0; j < image.getHeight(); j++) {
				Color c2 = new Color(image.getRGB(i, j));
				if (c.equals(c2))
					result++;
			}

		return result;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
//		File f = new File("/home/mehran/Desktop/brodatz_resized");
//		for (File file: f.listFiles()) {
//			batchCalculateAvgSd("/home/mehran/Desktop/brodatz_resized/", file.getName().split("\\.")[0], "bmp");
//		}
			
		
//		if (true)
//			return;
		
		String filePath = "textures/second-try/";
		String fileName = "test";
		String format = "png";
		String ext = "." + format;
		
		int x = 15, y = 15, step = 2, loopCount = 3;

		try {
			BufferedImage img = ImageIO.read(new File(filePath + fileName + ext));

			System.out.println(countPixels(new Color(0, 255, 0), img));

			BufferedImage imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());

			for (int a = 0; a < loopCount; a++) {
				img = ImageIO.read(new File(filePath + fileName + ext));
				imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
				System.out.println("Generating " + x + "x" + y + " Average");

				for (int i = 0; i < img.getWidth(); i++)
					for (int j = 0; j < img.getHeight(); j++)
						imgAvg.setRGB(i, j, ImageUtils.meanRGB(img, i, j, x, y).getRGB());

				ImageIO.write(imgAvg, format, new File(filePath + fileName + "_avg" + (x - 12) + "x" + (y - 12)  + ext));

				img = ImageIO.read(new File(filePath + fileName + ext));
				imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());

				System.out.println("Generating " + x + "x" + y + " Standard Deviation");

				for (int i = 0; i < img.getWidth(); i++)
					for (int j = 0; j < img.getHeight(); j++)
						imgAvg.setRGB(i, j, ImageUtils.standardDeviation(img, i, j, x, y).getRGB());

				ImageIO.write(imgAvg, format, new File(filePath + fileName + "_sd" + (x - 12) + "x" + (y - 12) + ext));

				x += step;
				y += step;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void batchCalculateAvgSd(String filePath, String fileName, String format) {
		String ext = "." + format;
		
		int x = 15, y = 15, step = 2, loopCount = 3;

		try {
			System.out.println();
			System.out.println("Reading " + fileName);
			BufferedImage img = ImageIO.read(new File(filePath + fileName + ext));

			System.out.println(countPixels(new Color(0, 255, 0), img));

			BufferedImage imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());

			for (int a = 0; a < loopCount; a++) {
				img = ImageIO.read(new File(filePath + fileName + ext));
				imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
				System.out.println("Generating " + x + "x" + y + " Average");

				for (int i = 0; i < img.getWidth(); i++)
					for (int j = 0; j < img.getHeight(); j++)
						imgAvg.setRGB(i, j, ImageUtils.meanRGB(img, i, j, x, y).getRGB());

				ImageIO.write(imgAvg, format, new File(filePath + fileName + "_avg" + (x - 12) + "x" + (y - 12)  + ext));

				img = ImageIO.read(new File(filePath + fileName + ext));
				imgAvg = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());

				System.out.println("Generating " + x + "x" + y + " Standard Deviation");

				for (int i = 0; i < img.getWidth(); i++)
					for (int j = 0; j < img.getHeight(); j++)
						imgAvg.setRGB(i, j, ImageUtils.standardDeviation(img, i, j, x, y).getRGB());

				ImageIO.write(imgAvg, format, new File(filePath + fileName + "_sd" + (x - 12) + "x" + (y - 12) + ext));

				x += step;
				y += step;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}