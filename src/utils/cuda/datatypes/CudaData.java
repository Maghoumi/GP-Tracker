package utils.cuda.datatypes;

import static cuda.CudaInterop.*;
import static jcuda.driver.JCudaDriver.*;
import gp.datatypes.DataInstance;

import java.util.List;

import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public class CudaData {
	
	public int imageWidth;		// The width of the image currently stored in this instance of the object
	public int imageHeight;	// The height of the image currently stored in this instance of the object
	
	public CUdeviceptr dev_input = null;
	public CUdeviceptr dev_smallAvg = null;
	public CUdeviceptr dev_mediumAvg = null;
	public CUdeviceptr dev_largeAvg = null;
	public CUdeviceptr dev_smallSd = null;
	public CUdeviceptr dev_mediumSd = null;
	public CUdeviceptr dev_largeSd = null;
	public CUdeviceptr dev_labels = null;
	public CUdeviceptr dev_output = null;
	
	public float[] input;
	
	public float[] smallAvg;
	public float[] mediumAvg;
	public float[] largeAvg;
	
	public float[] smallSd;
	public float[] mediumSd;
	public float[] largeSd;
	
	public int[] labels;
	
	public byte[] output;
	
	/** Indicates the pitch of the float pointers if this object is allocated
	 * using pitched memory allocation */
	int floatPitch;
	
	public CudaData() {
		super();
	}
	
	/**
	 * Creates an instance of this class for training purposes.
	 * NOTE: A call to this function will automatically transfer the data to GPU memory
	 * 
	 * @param input
	 * @param smallAvg
	 * @param mediumAvg
	 * @param largeAvg
	 * @param smallSd
	 * @param mediumSd
	 * @param largeSd
	 * @param labels
	 * @param imageWidth
	 * @param imageHeight
	 * @param positiveCount
	 * @param negativeCount
	 */
	public CudaData(List<DataInstance> instances) {
		int count = instances.size();
		// Initialize the arrays
		this.input = new float[count * 4];
		this.smallAvg = new float[count * 4];
		this.mediumAvg = new float[count * 4];
		this.largeAvg = new float[count * 4];
		this.smallSd = new float[count * 4];
		this.mediumSd = new float[count * 4];
		this.largeSd = new float[count * 4];
		this.labels = new int[count];
		
		int dataIndex = 0; 
		
		for (int i = 0 ; i < count ; i++) {
			DataInstance instance = instances.get(i);
			putInterleaved(this.input, instance.input, dataIndex);
			putInterleaved(this.smallAvg, instance.smallAvg, dataIndex);
			putInterleaved(this.mediumAvg, instance.mediumAvg, dataIndex);
			putInterleaved(this.largeAvg, instance.largeAvg, dataIndex);
			putInterleaved(this.smallSd, instance.smallSd, dataIndex);
			putInterleaved(this.mediumSd, instance.mediumSd, dataIndex);
			dataIndex = putInterleaved(this.largeSd, instance.largeSd, dataIndex); // and move the index forward
			
			this.labels[i] = instance.label;			
		}
		
		allocAndTransToMemory();
	}
	
	// Puts interleaved and returns the final index (the index to the beginning of the next element)
	private int putInterleaved(float[] inputArray, Float4 inputData, int index) {
		inputArray[index++] = inputData.x;
		inputArray[index++] = inputData.y;
		inputArray[index++] = inputData.z;
		inputArray[index++] = inputData.w;
		
		return index;
	}
	
	/**
	 * For when this class is used for training data, the filters and everything is pre-calculated.
	 * We just need to store the training data and transfer them to memory.
	 *
	 * @param input
	 * @param smallAvg
	 * @param mediumAvg
	 * @param largeAvg
	 * @param smallSd
	 * @param mediumSd
	 * @param largeSd
	 * @param labels
	 */
	public CudaData(float[] input,
			float[] smallAvg, float[] mediumAvg, float[] largeAvg,
			float[] smallSd, float[] mediumSd, float[] largeSd,
			int[] labels,
			int imageWidth, int imageHeight, int positiveCount, int negativeCount) {
		
		this.input = input;
		
		this.smallAvg = smallAvg;
		this.mediumAvg = mediumAvg;
		this.largeAvg = largeAvg;
		
		this.smallSd = smallSd;
		this.mediumSd = mediumSd;
		this.largeSd = largeSd;
		
		this.labels = labels;
		
		this.imageWidth = imageWidth;
		this.imageHeight = imageHeight;
		
		allocAndTransToMemory();
	}
	
	public void initOutput() {
		int outputBytes = imageWidth * imageHeight * 4;
		this.dev_output = new CUdeviceptr();
		cuMemAlloc(dev_output, outputBytes * Sizeof.BYTE);
	}
	
	/**
	 * Allocates all the required memory on CUDA's memory and
	 * transfers the fields to the memory
	 */
	public void allocAndTransToMemory() {
		// allocate input array and transfer
		if (input != null)
			dev_input = allocTransFloat(input);
		
		if (smallAvg != null)
			dev_smallAvg = allocTransFloat(smallAvg);
		if (mediumAvg != null)
			dev_mediumAvg = allocTransFloat(mediumAvg);
		if (largeAvg != null)
			dev_largeAvg = allocTransFloat(largeAvg);
		
		if (smallSd != null)
			dev_smallSd = allocTransFloat(smallSd);
		if (mediumSd != null)
			dev_mediumSd = allocTransFloat(mediumSd);
		if (largeSd != null)
			dev_largeSd = allocTransFloat(largeSd);
		
		if (labels != null)
			dev_labels = allocTransInt(labels);
		
		if(output != null)
			dev_output = allocTransByte(output);
	}
	
	
	
	/**
	 * Frees all allocated memory on the GPU that the instance of this class
	 * is currently using
	 */
	public void freeAll() {
		if (input != null)
			cuMemFree(dev_input);
		
		if (smallAvg != null)
			cuMemFree(dev_smallAvg);
		if (mediumAvg != null)
			cuMemFree(dev_mediumAvg);
		if (largeAvg != null)
			cuMemFree(dev_largeAvg);
		
		if (smallSd != null)
			cuMemFree(dev_smallSd);
		if (mediumSd != null)
			cuMemFree(dev_mediumSd);
		if (largeSd != null)
			cuMemFree(dev_largeSd);
		
		if (labels != null)
			cuMemFree(dev_labels);
		
		if(output != null)
			cuMemFree(dev_output);
	}
	
	/**
	 * Light clones this instance of the class. All array data are 
	 * cloned but device pointers are remained intact.
	 * 
	 * @return
	 */
	public CudaData lightClone() {
		CudaData result = new CudaData();
		result.input = this.input.clone();
		
		result.smallAvg = smallAvg.clone();
		result.mediumAvg = mediumAvg.clone();
		result.largeAvg = largeAvg.clone();
		
		result.smallSd = smallSd.clone();
		result.mediumSd = mediumSd.clone();
		result.largeSd = largeSd.clone();
		
		//FIXME
		//		result.labels = labels.clone();
		
		result.imageWidth = imageWidth;
		result.imageHeight = imageHeight;
		
		return result;
	}
}
