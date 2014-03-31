package utils.cuda.datatypes;

/**
 * Represents the box filter results obtained using CUDA
 * NOTE: the underlying representation is probably ABGR!
 * 
 * @author Mehran Maghoumi
 *
 */
public class FilterResult {
	public float[] avgFilter;
	public float[] sdFilter;
	
	public FilterResult(float[] avgFilter, float[] sdFilter) {
		this.avgFilter = avgFilter;
		this.sdFilter = sdFilter;
	}
}
