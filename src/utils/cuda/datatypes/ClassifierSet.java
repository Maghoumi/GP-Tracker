package utils.cuda.datatypes;

import java.awt.Color;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;

import cuda.CudaInterop;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;


/**
 * A HashSet of classifiers but with utility methods for transferring all
 * individuals to the GPU memory. I have tried to make this set thread-safe.
 * The add/remove/update/getPointerToAll methods are all thread-safe.
 * 
 * @author Mehran Maghoumi
 *
 */
public class ClassifierSet extends TreeSet<Classifier> {
	
	/** For synchronization purposes */
	private Object mutex = new Object();
	
	/** The number of enabled classifiers in this set */
	private int enabledSize = 0;
	
	/** The maximum length of the classifier currently present in this set */
	private int maxExpLength = 0;
	
	@Override
	public synchronized boolean add(Classifier e) {
		synchronized (mutex) {
			boolean result = super.add(e);
			e.setOwner(this);
				
			if (e.isEnabled()) { // If the classifier is enabled, it's size should be considered for maximum expression length
				enabledSize++;
				// Should update the maximum expression length if necessary 
				int expLength = e.getExpression().length;
				
				if (expLength > maxExpLength)
					maxExpLength = expLength;
			}
				
			return result;
		}
	}
	
	/**
	 * Updates a classifier that already exists in this list
	 * @param e	The new classifier
	 */
	public synchronized void update(Classifier e) {
		synchronized (mutex) {
			remove(e);
			add(e);
		}
	}
	
	@Override
	public synchronized boolean remove(Object o) {
		synchronized (mutex) {
			Classifier e = (Classifier) o;
			boolean result = super.remove(o);
			
			if (!result)	// if this classifier was not in the set in the first place, we shouldn't do anything else
				return result;
			
			e.setOwner(null);
			
			// Update maximum expression length if necessary
			if (e.isEnabled()) {
				
				enabledSize--;				
				int length = ((Classifier)o).getExpression().length;
				
				if (length == this.maxExpLength) {	// Should scan through all elements
					this.maxExpLength = getNewMaxLength();
				}
			}
			
			return result;
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
	
	/**
	 * @return Returns the number of enabled classifier in this set.
	 */
	public int getEnabledSize() {
		return this.enabledSize;
	}
	
	/**
	 * Notify this set that the enable status of a classifier has changed
	 * @param e
	 */
	public void notifyEnabilityChanged(Classifier e) {
		if (this.contains(e)) {
			this.update(e);
		}
	}
	
	
	/**
	 * Transfers all expressions to the GPU as a pitched memory and 
	 * returns the pointer to the allocated space. The disabled classifiers are
	 * simply ignored as if they never existed
	 * 
	 * @return
	 */
	public synchronized ClassifierAllocationResult getPointerToAll() {
		synchronized (mutex) {
			if (this.getEnabledSize() <= 0)	// Nothing to do if we don't have anything
				return null;
			
			byte[] expressions = new byte[this.getEnabledSize() * maxExpLength];	// Holder for 2D host expression array
			byte[] overlayColors = new byte[this.getEnabledSize() * 4]; // Holder for the overlay colors
			
			int expOffsetIdx = 0 ; // The offset index for the expression array
			int overlayOffsetIdx = 0; // The offset index for the 
			List<Classifier> activeClassifiers = new ArrayList<Classifier>();	// The classifiers that are enabled and active 
			
			for (Classifier classifier : this) {
				if (!classifier.isEnabled())	// If this classifier is not enabled, don't event bother doing anything for it
					continue;
				
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
			}
			
			CUdeviceptr2D expResult = new CUdeviceptr2D(maxExpLength, this.getEnabledSize(), 1, Sizeof.BYTE);	// Device 2D array
			expResult.allocTransByte(expressions);
			CUdeviceptr overlayResult = CudaInterop.allocTransByte(overlayColors);
			
			return new ClassifierAllocationResult(activeClassifiers, expResult, overlayResult);
		}
	}
	
	public class ClassifierAllocationResult {
		public CUdeviceptr2D expressions;
		public CUdeviceptr overlayColors;
		public List<Classifier> classifiers;
		
		public ClassifierAllocationResult(List<Classifier> classifiers, CUdeviceptr2D expressions, CUdeviceptr overlayColors) {
			this.classifiers = classifiers;
			this.expressions = expressions;
			this.overlayColors = overlayColors;
		}
	}
	
}
