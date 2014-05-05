package utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

import utils.cuda.pointers.CudaByte2D;

/**
 * A TreeSet of classifiers but with utility methods for transferring all
 * individuals to the GPU memory. I have tried to make this set thread-safe.
 * The add/remove/update/getPointerToAll methods are all thread-safe.
 * 
 * @author Mehran Maghoumi
 *
 */
public class ClassifierSet implements Iterable<Classifier> {
	
	/**
	 * For synchronization purposes: when the classifiers are being converted for CUDA,
	 * no other classifiers must be modified, otherwise a ConcurrentModificationException
	 * will be thrown!
	 */
	private Object mutex = new Object();
	
	private TreeSet<Classifier> set = new TreeSet<>();
	
	public boolean add(Classifier e) {
		synchronized (mutex) {
			return set.add(e);
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
	
	public boolean remove(Object o) {
		synchronized (mutex) {
			return set.remove(o);
		}
	}
	
	public boolean isEmpty() {
		return set.isEmpty();
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
			if (isEmpty())	// Nothing to do if we don't have anything
				return null;
			
			int maxExpLength = 0;
			List<Classifier> activeClassifiers = new ArrayList<Classifier>();	// The classifiers that are enabled and active
			
			byte[][] clasfExps = new byte[set.size()][];
			int idxMaintain = 0;
			int overlayOffsetIdx = 0; // The offset index for the overlay colors
			int enMapIndex = 0;	// The offset index for the enability map
			byte[] overlayColors = new byte[set.size() * 4]; // Holder for the overlay colors
			byte[] enabilityMap = new byte[set.size()]; // Holder for the overlay colors
			
			// We will maintain another set of Classifier expressions.
			// Although we are locking the set, the Classifiers in the set
			// could still be changed by the GPEngine causing an
			// ArrayIndexOutOfBoundsException in System.arraycopy
			
			// Stored the cloned expressions and do some other stuff (determine maxLength, enability, etc.)
			for (Classifier classifier : this.set) {
				classifier.resetClaims();	// reset this guy's claims
				// Add this classifier to the list of active classifiers
				activeClassifiers.add(classifier);
				
				Color c = classifier.getColor();
				overlayColors[overlayOffsetIdx++] = (byte) c.getAlpha();
				overlayColors[overlayOffsetIdx++] = (byte) c.getBlue();
				overlayColors[overlayOffsetIdx++] = (byte) c.getGreen();
				overlayColors[overlayOffsetIdx++] = (byte) c.getRed();
				
				enabilityMap[enMapIndex++] = (byte) (classifier.isEnabled() ? 1 : 0);
				
				clasfExps[idxMaintain] = classifier.getClonedExpression();
				
				if (clasfExps[idxMaintain].length > maxExpLength)
					maxExpLength = clasfExps[idxMaintain].length;
				
				idxMaintain++;
			}
			
			byte[] expressions = new byte[set.size() * maxExpLength];	// Holder for 2D host expression array
			int expOffsetIdx = 0 ; // The offset index for the expression array
			
			for (byte[] expression : clasfExps) {
				
				try {
					System.arraycopy(expression, 0, expressions, expOffsetIdx * maxExpLength, expression.length);	// Copy the expression into the correct offset
				}
				catch(Throwable t) {
					String debug = "exppression.length=" + expression.length + " " +
									"expressions.length=" + expressions.length + " " +
									"expOffsetIdx * maxExpLength=" + expOffsetIdx * maxExpLength + " " +
									"set.size()=" + set.size();
					System.err.println(debug);
					throw new RuntimeException(t);
				}

				expOffsetIdx++;
				
			}
			
			CudaByte2D expResult = new CudaByte2D(maxExpLength, set.size(), 1, expressions);
			CudaByte2D overlayResult = new CudaByte2D(set.size(), 1, 4, overlayColors);
			CudaByte2D enabilityResult = new CudaByte2D(set.size(), 1, 1, enabilityMap);
			
			return new ClassifierAllocationResult(activeClassifiers, expResult, overlayResult, enabilityResult);
		}
	}
	
	/**
	 * @return	Returns the length of the longest GP-expression that currently
	 * 			exists in this set.
	 */
	private int getMaxExpLength() {
		int result = 0;
		
		for (Classifier c : this.set) {
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

	@Override
	public Iterator<Classifier> iterator() {
		synchronized (mutex) {
			return this.set.iterator();
		}
	}
	
}
