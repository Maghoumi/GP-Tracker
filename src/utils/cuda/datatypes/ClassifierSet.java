package utils.cuda.datatypes;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import utils.cuda.datatypes.pointers.CudaByte2D;

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
	
	@Override
	public boolean add(Classifier e) {
		synchronized (mutex) {
			return super.add(e);
		}
	}
	
	/**
	 * Updates a classifier that already exists in this list
	 * @param e	The new classifier
	 */
	public boolean update(Classifier e) {
		synchronized (mutex) {
			remove(e);
			boolean result = add(e);
			return result;
		}
	}
	
	@Override
	public boolean remove(Object o) {
		synchronized (mutex) {
			return super.remove(o);
		}
	}
	
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
			
			int maxExpLength = getMaxExpLength();
			
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
			
			CudaByte2D expResult = new CudaByte2D(maxExpLength, this.size(), 1, expressions);
			CudaByte2D overlayResult = new CudaByte2D(this.size(), 1, 4, overlayColors);
			CudaByte2D enabilityResult = new CudaByte2D(this.size(), 1, 1, enabilityMap);
			
//			CUdeviceptr2D expResult = new CUdeviceptr2D(maxExpLength, this.size(), 1, Sizeof.BYTE);	// Device 2D array
//			expResult.allocTransByte(expressions);
//			CUdeviceptr overlayResult = CudaInterop.allocTransByte(overlayColors);
//			CUdeviceptr enabilityResult = CudaInterop.allocTransByte(enabilityMap);
			
			return new ClassifierAllocationResult(activeClassifiers, expResult, overlayResult, enabilityResult);
		}
	}
	
	/**
	 * @return	Returns the length of the longest GP-expression that currently
	 * 			exists in this set.
	 */
	private int getMaxExpLength() {
		int result = Integer.MIN_VALUE;
		
		for (Classifier c : this) {
			if (c.getExpression().length > result)
				result = c.getExpression().length;
		}
		
		return result;
	}

	public class ClassifierAllocationResult {
		/** The allocated expression trees of these classifier */
		public CudaByte2D expressions;
		
		/** The overlay color of each classifier */
		public CudaByte2D overlayColors;
		
		/** For each classifier: is it enabled or not? */
		public CudaByte2D enabilityMap;
		
		/** Just a list containing the classifiers in this set */
		public List<Classifier> classifiers;
		
		public ClassifierAllocationResult(List<Classifier> classifiers, CudaByte2D expressions, CudaByte2D overlayColors, CudaByte2D enabilityMap) {
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
			overlayColors.free();
			enabilityMap.free();
		}
	}
	
}
