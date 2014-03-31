package utils;

public class CollectionUtils
{

	public static void AddMatrices(double[][] a, double[][] b)
	{
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[i].length; j++)
				a[i][j] = a[i][j] + b[i][j];
	}
	
	public static void SubMatrices(double[][] a, double[][] b)
	{
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[i].length; j++)
				a[i][j] = a[i][j] - b[i][j];
	}
	
	public static void MulMatrices(double[][] a, double[][] b)
	{
		double[][] result = new double[a.length][a[0].length];
		
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[0].length; j++)
				for (int k = 0; k < b.length; j++)
				a[i][j] = a[i][j] - b[i][j];
	}

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{

	}

}
