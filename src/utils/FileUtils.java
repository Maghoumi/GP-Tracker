package utils;

import java.awt.Point;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class FileUtils
{
	public void readInputFile(String path, int size, String unparsedData) throws Exception
	{
		BufferedReader bf = new BufferedReader(new InputStreamReader(new FileInputStream(path)));

		//TODO Clear output collection etc.

		String line = "";

		// sentinel variable
		int i = 0;

		while (i < size && (line = bf.readLine()) != null)
		{
			String[] splitted = line.split(","); // split the input based on commas

			// store read values in the data array
			for (int j = 0; j < 15; j++)
			{
				//ed.data[j] = Double.parseDouble(splitted[j]);
			}

			// load parsed input into memory
			//inputData.add(ed);
			i++;
		}

		bf.close();

		if (unparsedData != null) // discards the rows containing "?"
		{
			BufferedReader bff = new BufferedReader(new InputStreamReader(new FileInputStream(unparsedData)));
			line = "";
			int counter = 0;

			while ((line = bff.readLine()) != null)
			{
				if (line.contains("?"))
					//inputData.remove(counter);

					counter++;
			}

			bff.close();
		}

		//TODO Shuffle!
	}

	public static int readTrainingPointsFromFile(ArrayList<Point> points, String path) throws Exception
	{
		BufferedReader bf = new BufferedReader(new FileReader(path));
		String line;
		int count = 0;
		
		while ((line = bf.readLine()) != null)
		{
			String[] splitted = line.split(",");
			Point p = new Point(Integer.parseInt(splitted[0]), Integer.parseInt(splitted[1]));
			
			points.add(p);
			count++;
		}
		
		bf.close();
		
		return count;
	}

	/**
	 * Appends the given text to the given file
	 * 
	 * @param text
	 *            Text to append
	 * @param path
	 *            Path to the file
	 * @throws Exception
	 */
	public static void wholeDump(String text, String path) throws Exception
	{
		BufferedWriter bw = new BufferedWriter(new FileWriter(path, true));
		bw.write(text);
		bw.flush();
		bw.close();
	}

	public static void main(String[] args)
	{
		// TODO Auto-generated method stub

	}

}
