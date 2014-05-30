package gp.datatypes;


import java.util.List;

import utils.cuda.pointers.CudaFloat2D;
import utils.cuda.pointers.CudaInteger2D;

/**
 * Represents a list of TrainingInstance that have been copied to the GPU memory
 * to be used in the CUDA evaluation kernel.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaTrainingInstance {
	
	/** Float2D object to hold the input RGB values */
	protected CudaFloat2D inputs;
	
	/** Float2D object to hold the smallAvg RGB values */
	protected CudaFloat2D smallAvgs;
	
	/** Float2D object to hold the mediumAvg RGB values */
	protected CudaFloat2D mediumAvgs;
	
	/** Float2D object to hold the largeAvg RGB values */
	protected CudaFloat2D largeAvgs;
	
	/** Float2D object to hold the smallSd RGB values */
	protected CudaFloat2D smallSds;
	
	/** Float2D object to hold the mediumSd RGB values */
	protected CudaFloat2D mediumSds;
	
	/** Float2D object to hold the largeSd RGB values */
	protected CudaFloat2D largeSds;
	 
	/** Integer2D object to hold the label of each instance */
	protected CudaInteger2D labels;
	
	/** Float2D object to hold the output of each instance */
	protected CudaFloat2D outputs;
	
	/** The width of the CudaPrimitive2D objects to allocate */
	protected int width;
	
	/** The number of channels per pixel */
	protected int numChannels;
	
	public CudaTrainingInstance(List<TrainingInstance> instances, int numChannels) {
		this.width = instances.size();
		this.numChannels = numChannels; 
		int byteCount = width * numChannels;
		
		// Create destination host arrays
		float[] inputs = new float[byteCount];
		float[] smallAvgs = new float[byteCount];
		float[] mediumAvgs = new float[byteCount];
		float[] largeAvgs = new float[byteCount];
		float[] smallSds = new float[byteCount];
		float[] mediumSds = new float[byteCount];
		float[] largeSds = new float[byteCount];
		int[] labels = new int[instances.size()];
		
		int floatCounter = 0;
		int intCounter = 0;
		
		// Copy training data to host arrays
		for (TrainingInstance instance : instances) {
			fillFloat(inputs, instance.input, floatCounter, numChannels);
			fillFloat(smallAvgs, instance.smallAvg, floatCounter, numChannels);
			fillFloat(mediumAvgs, instance.mediumAvg, floatCounter, numChannels);
			fillFloat(largeAvgs, instance.largeAvg, floatCounter, numChannels);
			fillFloat(smallSds, instance.smallSd, floatCounter, numChannels);
			fillFloat(mediumSds, instance.mediumSd, floatCounter, numChannels);
			floatCounter = fillFloat(largeSds, instance.largeSd, floatCounter, numChannels);
			intCounter = fillInt(labels, instance.label, intCounter);
		}
		
		// Allocate and transfer CUDA data
		this.inputs = new CudaFloat2D(width, 1, numChannels, inputs, true);
		
		this.smallAvgs = new CudaFloat2D(width, 1, numChannels, smallAvgs, true);
		this.mediumAvgs = new CudaFloat2D(width, 1, numChannels, mediumAvgs, true);
		this.largeAvgs = new CudaFloat2D(width, 1, numChannels, largeAvgs, true);
		
		this.smallSds = new CudaFloat2D(width, 1, numChannels, smallSds, true);
		this.mediumSds = new CudaFloat2D(width, 1, numChannels, mediumSds, true);
		this.largeSds = new CudaFloat2D(width, 1, numChannels, largeSds, true);
		
		this.labels = new CudaInteger2D(width, 1, 1, labels, true);
		this.outputs = new CudaFloat2D(width, 1, 1, true);
	}
	
	/**
	 * Allocates and transfers the fields to GPU memory using the calling thread's CUDA context
	 */
	public void allocateAndTransfer() {
		this.inputs.reallocate();
		
		this.smallAvgs.reallocate();
		this.mediumAvgs.reallocate();
		this.largeAvgs.reallocate();
		
		this.smallSds.reallocate();
		this.mediumSds.reallocate();
		this.largeSds.reallocate();
		
		this.labels.reallocate();
		this.outputs.reallocate();
	}

	/**
	 * Copies <i>numChannel</i> elements of the source array to the destination array at the specified
	 * index. The index is then incremented and returned.
	 *  
	 * @param destination
	 * @param source
	 * @param floatCounter
	 * @param numChannels
	 * @return
	 */
	private int fillFloat(float[] destination, float[] source, int floatCounter, int numChannels) {
		System.arraycopy(source, 0, destination, floatCounter, numChannels);
		floatCounter += numChannels;
		return floatCounter;
	}
	
	/**
	 * Copies the provided input to the destination array at the specified
	 * index. The index is then incremented and returned.
	 *  
	 * @param destination
	 * @param source
	 * @param floatCounter
	 * @param numChannels
	 * @return
	 */
	private int fillInt(int[] labels, int label, int intCounter) {
		labels[intCounter++] = label;
		return intCounter;
	}
	
	
	/**
	 * @return the inputs
	 */
	public CudaFloat2D getInputs() {
		return inputs;
	}

	/**
	 * @return the smallAvgs
	 */
	public CudaFloat2D getSmallAvgs() {
		return smallAvgs;
	}

	/**
	 * @return the mediumAvgs
	 */
	public CudaFloat2D getMediumAvgs() {
		return mediumAvgs;
	}

	/**
	 * @return the largeAvgs
	 */
	public CudaFloat2D getLargeAvgs() {
		return largeAvgs;
	}

	/**
	 * @return the smallSds
	 */
	public CudaFloat2D getSmallSds() {
		return smallSds;
	}

	/**
	 * @return the mediumSds
	 */
	public CudaFloat2D getMediumSds() {
		return mediumSds;
	}

	/**
	 * @return the largeSds
	 */
	public CudaFloat2D getLargeSds() {
		return largeSds;
	}

	/**
	 * @return the labels
	 */
	public CudaInteger2D getLabels() {
		return labels;
	}

	/**
	 * @return the outputs
	 */
	public CudaFloat2D getOutputs() {
		return outputs;
	}

	/**
	 * Frees all CUDA allocated memories that are associated with this object
	 */
	public void freeAll() {
		inputs.free();
		smallAvgs.free();
		mediumAvgs.free();
		largeAvgs.free();
		smallSds.free();
		mediumSds.free();
		largeSds.free();
		labels.free();
		outputs.free();
	}
	
}

