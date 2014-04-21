package gp.datatypes;


import utils.cuda.datatypes.*;

public class DataInstance {
	public int label;
	
	public Float4 input;
	
	public Float4 smallAvg;
	public Float4 mediumAvg;
	public Float4 largeAvg;
	
	public Float4 smallSd;
	public Float4 mediumSd;
	public Float4 largeSd;
	
	// not a copy constructor
	public DataInstance(Float4 input,
			Float4 smallAvg, Float4 mediumAvg, Float4 largeAvg,
			Float4 smallSd, Float4 mediumSd, Float4 largeSd,
			int label) {
		
		this.input = input;
		this.smallAvg = smallAvg;
		this.mediumAvg = mediumAvg;
		this.largeAvg = largeAvg;
		
		this.smallSd = smallSd;
		this.mediumSd = mediumSd;
		this.largeSd = largeSd;
		
		this.label = label;		
	}
	
	// Copy constructor
	public DataInstance (DataInstance in) {
		this.input = in.input.clone();
		this.smallAvg = in.smallAvg.clone();
		this.mediumAvg = in.mediumAvg.clone();
		this.largeAvg = in.largeAvg.clone();
		
		this.smallSd = in.smallSd.clone();
		this.mediumSd = in.mediumSd.clone();
		this.largeSd = in.largeSd.clone();
		
		this.label = in.label;
	}
	
	
	public DataInstance clone() {
		return new DataInstance(this);
	}
}
