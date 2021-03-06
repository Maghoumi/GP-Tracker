package invoker;

import feeder.VideoFeeder;
import gp.GPEngine;
import gp.datatypes.Job;
import utils.Classifier;
import utils.ColorUtils;
import utils.EvolutionListener;
import utils.Segment;
import utils.SegmentedVideoFrame;
import utils.SuccessListener;
import visualizer.Visualizer;

/**
 * An Invoker is the missing link between the VideoFeeder, the Visualizer and the GPEngine.
 * Basically, the VideoFeeder has to pass its video frames to the Invoker object. The Invoker
 * object will then decide if the GPEngine needs to be called. If necessary, Invoker will
 * "invoke" the GPEngine and mediate the communications with it. The Invoker object should
 * also mediate the communications with the Visualizer. The Invoker is the one who supplies
 * the Visualizer with necessary visualization data.
 * Note that the Invoker will also have its own thread. This means that The VideoFeeder object
 * must be a passive object and should provide video frames only when asked (as opposed to
 * automatically).
 *  
 * @author Mehran Maghoumi
 *
 */
public abstract class Invoker implements EvolutionListener, Runnable, SuccessListener {
	
	/** How often should the GPEngine report back? */
	public static final int REPORT_FREQUENCY = 1;
	
	/** The VideoFeeder object that passes this class its video frames */
	protected VideoFeeder feeder;
	
	/** The GPEngine instance that this Invoker uses to evolve classifiers */
	protected GPEngine gpSystem;
	
	/** The Visualizer that this instance uses to visualize the evolution results */
	protected Visualizer visualizer;
	
	/**
	 * The worker thread of this Invoker that constantly obtains frames from the
	 * VideoFeeder object and passes data to the Visualizer
	 */
	protected Thread workerThread;
	
	/** Run flag of the worker thread */
	protected volatile boolean threadAlive = false;
	
	/** The prefix of this run session. Used for logging purposes only */
	protected String sessionPrefix;
	
	/**
	 * Instantiates an Invoker object and also initializes the GPEngine object
	 * that this instance requires and registers itself with the GPEngine as an
	 * evolution listener.
	 * 
	 * @param gpArgs	ECJ's command-line parameters
	 * @param sessionPrefix	The prefix of the dump files of the session
	 */
	public Invoker(String[] gpArgs, String sessionPrefix) {
		this.gpSystem = new GPEngine(gpArgs, false, sessionPrefix);
		this.sessionPrefix = sessionPrefix;
		gpSystem.addEvolutionListener(this);
		gpSystem.addSuccessListener(this);
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
			
			// We only obtain the frame and then pass it!
			// It's up to the Visualizer to let us know that there are problems
			// and that we should handle them			
			visualizer.passNewFrame(newFrame, 0);			
		}		
	}
	
	public boolean isQueueEmpty() {
		return gpSystem.isQueueEmpty();
	}
	
	/**
	 * Determines if the GP system has choked for this segment (could not
	 * find a classifier even though maximum number of GP calls reached)
	 * 
	 * @param s
	 * @return
	 */
	public boolean hasChoked(Segment s) {
		return this.gpSystem.hasReachedLimit(s.toString());
	}
	
	/**
	 * Evolves a classifier that can distinguish the target segment from
	 * other segments that exist in the provided SegmentedVideoFrame
	 * 
	 * @param frame		A frame containing various textures, including the target
	 * @param target	The target for positive classification
	 */
	public void evolveClassifier(SegmentedVideoFrame frame, Segment target) {
		if (!this.gpSystem.isQueueEmpty())
			return;
		
		Classifier classifier = new Classifier(Classifier.TYPE_POS_NEG);
		classifier.addPositiveExample(target.getByteImage());
		classifier.addNegativeExample(frame.getBackground().getByteImage());
		
		for (Segment otherSegment : frame) {
			if (!otherSegment.equals(target)) {
				classifier.addNegativeExample(otherSegment.getByteImage());	// Add other's image data as negative instances
			}
		}
		
		Job j = new Job(classifier, target.toString());
		j.setTag(this.visualizer);
		gpSystem.queueJob(j);
	}
	
	/**
	 * Retrains the specified classifier
	 *  
	 * @param shouldSeed	Should the existing GPTree be used as the initial population seed?  
	 */
	public void retrain(Classifier classifier, boolean shouldSeed) {
		classifier.setShouldSeed(shouldSeed);
		gpSystem.queueJob(new Job(classifier, "__retrain"));
	}
	
	@Override
	public boolean reportClassifier(Classifier classifier) {
		return this.visualizer.passNewClassifier(classifier);
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
	 * Enables or disables the underlying GPEngine object of this Invoker.
	 * If the GPEngine is disabled, the jobs on the queue are not processed.
	 * @param status
	 */
	public void setGPStatus(boolean status) {
		if (status)
			this.gpSystem.startWorkerThread();
		else
			this.gpSystem.stopWorkerThread();
	}
}
