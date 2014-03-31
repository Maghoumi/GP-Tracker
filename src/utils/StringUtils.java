package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import org.omg.CORBA.Environment;

public class StringUtils {
	
	public static String TextToString(String path) throws Exception {
		StringBuilder sb = new StringBuilder("");
		
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(path))));
		String line;

		while ((line = br.readLine()) != null)
			sb.append(line + System.lineSeparator());

		br.close();
		
		return sb.toString();
	}
	
}
