package invoker;



import javax.swing.UIManager;

import utils.cuda.datatypes.Classifier;
import utils.cuda.datatypes.ClassifierSet;
import utils.cuda.datatypes.Segment;
import video_interop.VideoFeeder;
import video_interop.datatypes.SegmentedVideoFrame;
import visualizer.Visualizer;
import m2xfilter.GPSystem;
import m2xfilter.datatypes.EvolutionListener;
import m2xfilter.datatypes.Job;

/**
 * An Invoker is the missing link between the VideoFeeder, the Visualizer and the GPSystem.
 * Basically, the VideoFeeder has to pass its video frames to the Invoker object. The Invoker
 * object will then decide if the GPSystem needs to be called. If necessary, Invoker will
 * "invoke" the GPSystem and mediate the communications with it. The Invoker object should
 * also mediate the communications with the Visualizer. The Invoker is the one who supplies
 * the Visualizer with necessary visualization data.
 * Note that the Invoker will also have its own thread. This means that The VideoFeeder object
 * must be a passive object and should provide video frames only when asked (as opposed to
 * automatically).
 *  
 * @author Mehran Maghoumi
 *
 */
public abstract class Invoker implements EvolutionListener, Runnable {
	
	/** How often should the GPSystem report back? */
	public static final int REPORT_FREQUENCY = 1;
	
	/** The VideoFeeder object that passes this class its video frames */
	protected VideoFeeder feeder;
	
	/** The GPSystem instance that this Invoker uses to evolve classifiers */
	protected GPSystem gpSystem;
	
	/** The Visualizer that this instance uses to visualize the evolution results */
	protected Visualizer visualizer;
	
	/** The list of all evolved classifiers */
	protected ClassifierSet evolvedClassifiers = new ClassifierSet();
	
	/**
	 * The worker thread of this Invoker that constantly obtains frames from the
	 * VideoFeeder object and passes data to the Visualizer
	 */
	protected Thread workerThread;
	
	/** Run flag of the worker thread */
	protected volatile boolean threadAlive = false;
	
	/**
	 * Instantiates an Invoker object and also initializes the GPSystem object
	 * that this instance requires and registers itself with the GPSystem as an
	 * evolution listener.
	 * 
	 * @param gpArgs	ECJ's command-line parameters
	 */
	public Invoker(String[] gpArgs) {
		this.gpSystem = new GPSystem(gpArgs, false);
		gpSystem.addEvolutionListener(this);
		this.workerThread = new Thread(this);
	}
	
	/**
	 * Starts the worker thread of this Invoker instance.
	 */
	public void start() {
		this.threadAlive = this.workerThread.isAlive();
		
		if (!this.threadAlive) {
			this.workerThread.start();
			this.threadAlive = true;
		}
		
		gpSystem.startWorkerThread();
	}
	
	/**
	 * Stops the worker thread of this invoker instance.
	 */
	public void stop() {
		this.threadAlive = false;
	}
	
	@Override
	public void run() {
		
		while(this.threadAlive) {
			// Obtain the next frame from the VideoFeeder
			SegmentedVideoFrame newFrame = this.feeder.getNextSegmentedFrame();
			
			// TODO decide if GP should be invoked
			if (this.evolvedClassifiers.size() == 0 && this.gpSystem.isQueueEmpty())
				invokeGPFromScratch(newFrame);
			
			// pass the new frame to the visualizer for visualization
			visualizer.passNewFrame(newFrame, 0);
			
		}
		
	}
	
	/**
	 * Invokes the GPSystem for (probably) the first time when there are no
	 * evolved classifiers in the system.
	 * 
	 * @param frame	The frame containing segments that need classifiers
	 */
	protected void invokeGPFromScratch(SegmentedVideoFrame frame) {
		for(Segment segment : frame)
			evolveClassifier(frame, segment);
	}
	
	/**
	 * Evolves a classifier that can distinguish the target segment from
	 * other segments that exist in the provided SegmentedVideoFrame
	 * 
	 * @param frame		A frame containing various textures, including the target
	 * @param target	The target for positive classification
	 */
	public void evolveClassifier(SegmentedVideoFrame frame, Segment target) {
		Classifier classifier = new Classifier(Classifier.TYPE_POS_NEG);
		classifier.addPositiveExample(target.getByteImage());
		classifier.addNegativeExample(frame.getBackground().getByteImage());
		
		for (Segment otherSegment : frame) {
			if (!otherSegment.equals(target)) {
				classifier.addNegativeExample(otherSegment.getByteImage());	// Add other's image data as negative instances
			}
		}
		
		Job j = new Job(classifier);
		gpSystem.queueJob(j);
	}
	
	/**
	 * Retrains the specified classifier
	 *  
	 * @param shouldSeed	Should the existing GPTree be used as the initial population seed?  
	 */
	public void retrain(Classifier classifier, boolean shouldSeed) {
		classifier.setShouldSeed(shouldSeed);
		gpSystem.queueJob(new Job(classifier));
	}
	
	@Override
	public void reportClassifier(Classifier classifier) {
		this.evolvedClassifiers.add(classifier);
		this.visualizer.passNewClassifier(classifier);
	}

	@Override
	public int getIndReportingFrequency() {
		return REPORT_FREQUENCY;
	}
	
	/**
	 * @return	The width of the video frames that are passed by the underlying
	 * 		VideoFeeder object
	 */
	public int getFrameWidth() {
		return this.feeder.getFrameWidth();
	}
	
	/**
	 * @return	The height of the video frames that are passed by the underlying
	 * 		VideoFeeder object
	 */
	public int getFrameHeight() {
		return this.feeder.getFrameHeight();
	}
	
	/**
	 * Toggles the paused state of the underlying VideoFeeder.
	 * @return	The pause state after toggling it
	 */
	public boolean togglePaused() {
		return this.feeder.togglePaused();
	}
	
	/**
	 * Enables or disables the underlying GPSystem object of this Invoker.
	 * If the GPSystem is disabled, the jobs on the queue are not processed.
	 * @param status
	 */
	public void setGPStatus(boolean status) {
		if (status)
			this.gpSystem.startWorkerThread();
		else
			this.gpSystem.stopWorkerThread();
	}
}