package gp.datatypes;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;

/**
 * A utility class that keeps track of various object tracking statistics
 * in the system.
 * 
 * @author Mehran Maghoumi
 *
 */
public class TrackerStatistics {
	/** Keeps track of how many times a segment has been queued for retraining */
	protected HashMap<String, Integer> trainingRequests = new HashMap<>();
	
	/** The file to dump the statistics to */
	protected File dumpFile;
	
	public TrackerStatistics(String dumpPath) {
		this.dumpFile = new File(dumpPath);
	}
	
	/**
	 * Adds the provided Job to the statistics log
	 * @param j
	 */
	public void addToStat(Job j) {
		String id = j.getId();
		int newValue = 1;
		
		if (this.trainingRequests.containsKey(id)) {
			newValue = this.trainingRequests.get(id);
			newValue++;
		}
		
		this.trainingRequests.put(id, newValue);
		dumpToFile();
	}
	
	protected void dumpToFile() {
		StringBuilder result = new StringBuilder();
		
		for (Entry<String, Integer> entry : this.trainingRequests.entrySet()) {
			result.append(entry.getKey() + "\t" + entry.getValue() + System.lineSeparator());			
		}
		
		result.replace(result.length() - 2, result.length(), "");
		
		try {
			FileUtils.writeStringToFile(dumpFile, result.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}