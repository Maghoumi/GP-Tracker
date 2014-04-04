package m2xfilter.datatypes;

import utils.cuda.datatypes.Classifier;
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
	 * Report the evolution of a new classifier to this object.
	 * @param classifier	The newly evolved classifier
	 */
	public void reportClassifier(Classifier classifier);
	
	/**
	 * How often do you want to be informed of the evolution of a new individual?  
	 * @return	the reporting frequency
	 */
	public int getIndReportingFrequency();
}
