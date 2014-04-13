package video_interop;

import utils.cuda.datatypes.ByteImage;
import video_interop.datatypes.SegmentedVideoFrame;
import visualizer.OpenGLVisualizer;

/**
 * Defines the standard operations that must be supported by any class that 
 * wants to act as a video feeder and pass video data to the Visualizer. Any
 * class that wants to act as a video feeder must also have its own thread
 * which will parse video frames and automatically pass them to its owning 
 * Visualizer instance. 
 * 
 * @author Mehran Maghoumi
 *
 */
public interface VideoFeeder {
	
//	/**
//	 * Set the owner of this VideoFeeder instance
//	 * @param visualizer	The visualizer that owns this VideoFeeder instance
//	 */
//	public void setOwner(OpenGLVisualizer visualizer);
	
	/**
	 * @return	The width of each frame of the video
	 */
	public int getFrameWidth();
	
	/**
	 * @return	The height of each frame of the video
	 */
	public int getFrameHeight();	
	
	/**
	 * Obtains the next frame from the video source and moves the 
	 * current position to the next frame. The feed should be restarted 
	 * if the next position is greater than the length of the video 
	 * 
	 * @return	The next video frame as a ByteImage object
	 */
	public ByteImage getNextFrame();
	
	/**
	 * @return	Obtains the next segmented frame from the video source and moves the 
	 * 			current position to the next frame. The feed should be restarted 
	 * 			if the next position is greater than the length of the video 
	 */
	public SegmentedVideoFrame getNextSegmentedFrame();
	
//	/**
//	 * Gets the next frame from the video source and passes the new frame
//	 * and the discovered segments to the owning Visualizer instance. A call
//	 * to this function should pause the worker thread and cause the execution
//	 * to switch to step mode 
//	 */
//	public void getAndPassNextFrame();
	
	/**
	 * @return	The length of the video in frames
	 */
	public int getLengthInFrames();
	
	/**
	 * @return	The current position of the video
	 */
	public int getCurrentPosition();
	
	/**
	 * Set the current position of the video
	 * @param position	The new position to set
	 */
	public void setCurrentPosition(int position);	
	
//	/**
//	 * The Runnable interface implementation. The thread which runs this function
//	 * must be able to automatically parse the video frames, advance the video
//	 * position and pass the parsed frame (along with other necessary information
//	 * such as information about detected segments in the frame) to the owning
//	 * Visualizer instance. 
//	 */
//	public void run();
//	
//	/**
//	 * Starts the worker thread
//	 */
//	public void start();
//	
//	/**
//	 * Stops the worker thread
//	 */
//	public void stop();
	
	/**
	 * Restart the video
	 */
	public void restart();
	
	/**
	 * Pauses the video. After a call to this function, the worker thread must not
	 * pass additional video frames to the owning Visualizer.
	 */
	public void pause();
	
	/**
	 * Resumes the video. After a call to this function, the worker thread must continue
	 * passing video frames to the owning Visualizer.
	 */
	public void resume();
	
	/**
	 * Toggles the paused state of this VideoFeeder.
	 * @return	The pause state after toggling it
	 */
	public boolean togglePaused();
	
	/**
	 * @return	If this VideoFeeder is paused, true, otherwise false.
	 */
	public boolean isPaused();
}
