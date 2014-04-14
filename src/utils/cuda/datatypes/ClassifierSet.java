package utils.cuda.datatypes;

import java.awt.Color;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;

import cuda.CudaInterop;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

/**
 * A TreeSet of classifiers but with utility methods for transferring all
 * individuals to the GPU memory. I have tried to make this set thread-safe.
 * The add/remove/update/getPointerToAll methods are all thread-safe.
 * 
 * @author Mehran Maghoumi
 *
 */
public class ClassifierSet extends TreeSet<Classifier> {
	
	/**
	 * For synchronization purposes: when the classifiers are being converted for CUDA,
	 * no other classifiers must be modified, otherwise a ConcurrentModificationException
	 * will be thrown!
	 */
	private Object mutex = new Object();
	
	/** The number of enabled classifiers in this set */
	
	/** The maximum length of the classifier currently present in this set */
	private int maxExpLength = 0;
	
	@Override
	public boolean add(Classifier e) {
		synchronized (mutex) {
			int expLength = e.getExpression().length;
			
			if (expLength > maxExpLength)
				maxExpLength = expLength;
			
			return super.add(e);
		}
	}
	
	/**
	 * Updates a classifier that already exists in this list
	 * @param e	The new classifier
	 */
	public void update(Classifier e) {
		synchronized (mutex) {
			remove(e);
			add(e);
		}
	}
	
	@Override
	public boolean remove(Object o) {
		synchronized (mutex) {
			int length = ((Classifier)o).getExpression().length;
			boolean result = super.remove(o);

			// Update the maxLength if necessary
			if (length == this.maxExpLength)	
				this.maxExpLength = getNewMaxLength();
			
			return result;
		}
	}
	
//	public boolean containsClassifierForSegment(Segment s) {
//		synchronized (mutex) {
//			s.getByteImage();
//			
//			for (Classifier c : this) {
//				c.get
//			}
//		}		
//	}
	
	/**
	 * Transfers all expressions to the GPU as a pitched memory and 
	 * returns the pointer to the allocated space. The disabled classifiers are
	 * also converted, however their disability is noted in the enability map!
	 * Furthermore, when this function is called, the classifiers' claims are
	 * also reset.
	 * 
	 * @return
	 */
	public synchronized ClassifierAllocationResult getPointerToAll() {
		// mutex will ensure that no other classifiers will be added, removed or updated while
		// we are converting the existing ones to pointers!
		synchronized (mutex) { 
			if (this.size() <= 0)	// Nothing to do if we don't have anything
				return null;
			
			byte[] expressions = new byte[this.size() * maxExpLength];	// Holder for 2D host expression array
			byte[] overlayColors = new byte[this.size() * 4]; // Holder for the overlay colors
			byte[] enabilityMap = new byte[this.size()]; // Holder for the overlay colors
			
			int expOffsetIdx = 0 ; // The offset index for the expression array
			int overlayOffsetIdx = 0; // The offset index for the overlay colors 
			int enMapIndex = 0;	// The offset index for the enability map
			List<Classifier> activeClassifiers = new ArrayList<Classifier>();	// The classifiers that are enabled and active 
			
			for (Classifier classifier : this) {
				classifier.resetClaims();	// reset this guy's claims
				// Add this classifier to the list of active classifiers
				activeClassifiers.add(classifier);
				
				byte[] exp = classifier.getExpression();
				System.arraycopy(exp, 0, expressions, expOffsetIdx * maxExpLength, exp.length);	// Copy the expression into the correct offset
				expOffsetIdx++;
				
				Color c = classifier.getColor();
				overlayColors[overlayOffsetIdx++] = (byte) c.getAlpha();
				overlayColors[overlayOffsetIdx++] = (byte) c.getBlue();
				overlayColors[overlayOffsetIdx++] = (byte) c.getGreen();
				overlayColors[overlayOffsetIdx++] = (byte) c.getRed();
				
				enabilityMap[enMapIndex++] = (byte) (classifier.isEnabled() ? 1 : 0);
			}
			
			CUdeviceptr2D expResult = new CUdeviceptr2D(maxExpLength, this.size(), 1, Sizeof.BYTE);	// Device 2D array
			expResult.allocTransByte(expressions);
			CUdeviceptr overlayResult = CudaInterop.allocTransByte(overlayColors);
			CUdeviceptr enabilityResult = CudaInterop.allocTransByte(enabilityMap);
			
			return new ClassifierAllocationResult(activeClassifiers, expResult, overlayResult, enabilityResult);
		}
	}
	
	/**
	 * @return	Finds the maximum expression length of the classifiers
	 * currently present in this set.
	 */
	private int getNewMaxLength() {
		int maxLength = -1;

		for (Classifier e : this) {			
			int length = e.getExpression().length;

			if (length > maxLength)
				maxLength = length;
		}

		return maxLength;
	}
	
	public class ClassifierAllocationResult {
		/** The allocated expression trees of these classifier */
		public CUdeviceptr2D expressions;
		/** The overlay color of each classifier */
		public CUdeviceptr overlayColors;
		/** For each classifier: is it enabled or not? */
		public CUdeviceptr enabilityMap;
		/** Just a list containing the classifiers in this set */
		public List<Classifier> classifiers;
		
		public ClassifierAllocationResult(List<Classifier> classifiers, CUdeviceptr2D expressions, CUdeviceptr overlayColors, CUdeviceptr enabilityMap) {
			this.classifiers = classifiers;
			this.expressions = expressions;
			this.overlayColors = overlayColors;
			this.enabilityMap = enabilityMap;
		}
		
		/**
		 * Frees the allocated CUDA memory for this classifier
		 */
		public void freeAll() {
			expressions.free();
			cuMemFree(overlayColors);
			cuMemFree(enabilityMap);
		}
	}
	
}
