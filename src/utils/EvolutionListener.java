package utils;

import ec.gp.GPIndividual;

/**
 * Represents the interface for all classes that want to listen in on various
 * ECJ evolution events. One such event would be the evolution of a new best
 * individual. A class may want to be informed by the CudaEvolutionState object
 * about the evolution of a new best individual. Other events (TODO) could be
 * the completion of a retrain job.
 * 
 * Hopefully, more functionality will be added to this interface.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface EvolutionListener {
	
	/**
	 * Report the evolution of a new classifier to this object. This function will
	 * tell the caller whether the evolution must be stopped or not (for eager termination)
	 * @param classifier	The newly evolved classifier
	 * @return	True, if evolution should be continued (no eager termination)
	 * 			False, for eager termination and if the reported classifier meets the
	 * 			requirements
	 */
	public boolean reportClassifier(Classifier classifier);
	
	/**
	 * How often do you want to be informed of the evolution of a new individual?  
	 * @return	the reporting frequency
	 */
	public int getIndReportingFrequency();
}
