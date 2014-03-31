package utils.cuda;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 * Represents a 2D CUDA device pointer: A pointer and a pointer to pointer
 * @deprecated This class is crap because it does not benefit from pitched memory
 * @author Mehran Maghoumi
 */
public class OldCUdeviceptr2D extends CUdeviceptr {
	
	// the second pointer which the base pointer is going to point to
	public CUdeviceptr[] levelPointer;
	// the 2nd dimension of the 2D array
	private int height;
	
	public OldCUdeviceptr2D(int height) {
		super();
		this.levelPointer = new CUdeviceptr[height];
		this.height = height;
	}
	
	public OldCUdeviceptr2D(CUdeviceptr basePointer, CUdeviceptr[] levelPointer) {
		super(basePointer);
		this.levelPointer = levelPointer;
		this.height = levelPointer.length;
	}
	
	public void setLevelPointer (int i, CUdeviceptr pointer) {
		levelPointer[i] = pointer;
	}
	
	public void setLevelPointer (CUdeviceptr[] pointer) {
		levelPointer = pointer;
	}
	
	public int getHeight() {
		return this.height;
	} 
	
	public void free() {
		
		for (int i = 0 ; i < height ; i++) {
			JCudaDriver.cuMemFree(levelPointer[i]);
		}
		
		JCudaDriver.cuMemFree(this);
	}
}
