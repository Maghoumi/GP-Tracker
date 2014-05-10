package gp.datatypes;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;

import visualizer.Visualizer;

/**
 * A utility class that keeps track of various object tracking statistics
 * in the system.
 * 
 * @author Mehran Maghoumi
 *
 */
public class TrackerStatistics {
	/** Maximum number of trainings allowed for each segment */
	public static final int MAX_RETRAININGS_ALLOWED = 26;	// ==> because the first one is a training the subsequent ones are retrainings
	
	/** Keeps track of how many times a segment has been queued for retraining */
	protected HashMap<String, Integer> trainingRequests = new HashMap<>();
	
	/** The file to dump the GP call statistics to */
	protected File callDumpPath;
	
	/** The file to dump the framerate statistics to */
	protected File fpsDumpPath;
	
	public TrackerStatistics(String callDumpPath, String fpsDumpPath) {
		this.callDumpPath = new File(callDumpPath);
		this.fpsDumpPath = new File(fpsDumpPath);
		
		if (this.fpsDumpPath.exists())
			this.fpsDumpPath.delete();
	}
	
	/**
	 * Adds the provided Job to the statistics log
	 * @param j	The job to log
	 * @return True if this job does not exceed the specified maximum number of
	 * 			retrainings per job, False if the maximum number of GP calls for
	 * 			this run has exceeded
	 */
	public boolean addToStat(Job j) {
		synchronized(trainingRequests) {
			String id = j.getId();
			int newValue = 1;
			
			if (this.trainingRequests.containsKey(id)) {
				newValue = this.trainingRequests.get(id);
				newValue++;
			}
			
			if (newValue > MAX_RETRAININGS_ALLOWED)
				return false;
			
			this.trainingRequests.put(id, newValue);
			dumpToFile();
			return true;
		}
	}
	
	/**
	 * Determines if a specific segment has reached the maximum number of allowed
	 * training requests.
	 * 
	 * @param segmentId
	 * @return	True if this segment has reached the maximum number of allowed requests
	 * 			False otherwise
	 */
	public boolean hasReachedLimit(String segmentId) {
		synchronized(trainingRequests) {
			if (trainingRequests.containsKey(segmentId)) {
				return (trainingRequests.get(segmentId).intValue() + 1) > MAX_RETRAININGS_ALLOWED;
			}
			
			return false;
		}
	}
	
	protected void dumpToFile() {
		StringBuilder result = new StringBuilder();
		
		for (Entry<String, Integer> entry : this.trainingRequests.entrySet()) {
			result.append(entry.getKey() + "\t" + entry.getValue() + System.lineSeparator());			
		}
		
		result.replace(result.length() - 2, result.length(), "");
		
		try {
			FileUtils.writeStringToFile(callDumpPath, result.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void addFrameStat(Job newJob) {
		String result = newJob.getId() + "\t" + ((Visualizer)newJob.getTag()).getFramerate() + System.lineSeparator();
		try {
			FileUtils.writeStringToFile(fpsDumpPath, result, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}