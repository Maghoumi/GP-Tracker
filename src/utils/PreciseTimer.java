package utils;

import ec.util.Output;

/**
 * A helper class that will precisely time an event and will report the
 * time that event took in seconds.
 * 
 * @author Mehran Maghoumi
 *
 */
public class PreciseTimer {
	
	/** Starting time of the event */
	private long start;
	
	/** Ending time of the event */
	private long stop;
	
	/**
	 * Mark the start of the timing process
	 */
	public void start() {
		start = System.nanoTime();
	}
	
	/**
	 * Mark the end of the timing process
	 */
	public void stop() {
		stop = System.nanoTime();
	}
	
	/**
	 * Log the message about the time the event took 
	 * in the standard output. The time is reported in
	 * seconds.
	 * 
	 * @param startMessage
	 * 		The emblem of the message
	 */
	public void log(String startMessage) {
		System.out.println(String.format(startMessage + " took : %5.5fs\n", (stop-start) / 1e9));
	}
	
	/**
	 * Mark the end of the event and log the event in the standard
	 * output.
	 * 
	 * @param startMessage
	 * 		The emblem of the message
	 */
	public void stopAndLog(String startMessage) {
		stop();
		log(startMessage);
	}
	
	/**
	 * Mark the end of the event and log the event in ECJ's output
	 * object
	 * 
	 * @param output
	 * 		The ECJ's output object to write to
	 * @param startMessage
	 * 		The emblem of the message
	 */
	public void stopAndLog(Output output, String startMessage) {
		stop();
		output.message(String.format(startMessage + " took : %5.5fs\n", (stop-start) / 1e9));
	}
}
