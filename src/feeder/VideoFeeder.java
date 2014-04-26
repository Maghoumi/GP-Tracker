package feeder;

import feeder.datatypes.SegmentedVideoFrame;
import utils.cuda.datatypes.ByteImage;
import visualizer.GLVisualizer;

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
