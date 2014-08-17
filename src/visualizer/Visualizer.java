package visualizer;

import invoker.Invoker;
import utils.Classifier;
import utils.ImageFilterProvider;
import utils.SegmentedVideoFrame;
import utils.SuccessProvider;

/**
 * Defines the interface that is required for any object that wants to act as a 
 * data visualizer for this evolutionary system. A visualizer object must be able
 * to receive visualization data from an Invoker object and must display those data
 * using CUDA. It should also support seeking in video positions, enabling and
 * disabling GP system as well as other (TODO) functionalities. 
 * 
 * @author Mehran Maghoumi
 *
 */
public interface Visualizer extends SuccessProvider {
	
	/**
	 * Set the dimensions for the video data that this visualizer must visualize.
	 * 
	 * @param width	The video width (in pixels)
	 * @param height	The video height (in pixels)
	 */
	public void setVideoDimensions(int width, int height);
	
	/**
	 * Set the length of the video that this visualizer shows. This method will
	 * set the maximum value for the slider that shows the video position.
	 * 
	 * @param lengthInFrames
	 *            The video length in frames
	 */
	public void setVideoLength(int lengthInFrames);
	
	/**
	 * Used to pass a new individual to the visualizer. Usually, when a fitter
	 * individual is found in the run, this function should be called. Note
	 * that the calling thread will be blocked if the individual queue of this
	 * visualizer is full.
	 * 
	 * @param classifier	The new classifier that is found during the GP run
	 * @return	True, if evolution should be continued (no eager termination)
	 * 			False, for eager termination and if the reported classifier meets the
	 * 			requirements 
	 */
	public boolean passNewClassifier(Classifier classifier);	
	
	/**
	 * Used to pass a new video frame (image) to this visualizer. Usually should
	 * be called after grabbing a frame of the video. Note that the calling
	 * thread will be blocked if the frame queue of the visualizer is full
	 * 
	 * @param frame	The new frame that is obtained from the video/image
	 * @param currentPosition	The current position in the video (the current 
	 * 		frame number)
	 */
	public void passNewFrame(SegmentedVideoFrame frame, int currentPosition);
	
	/**
	 * @return	The ImageFilterProvider implementation that this visualizer uses to filter
	 * 			images and display them
	 */
	public ImageFilterProvider getImageFilterProvider();
	
	/**
	 * @return	The current visualization framerate
	 */
	public double getFramerate();
}
