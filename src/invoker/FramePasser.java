package invoker;

import java.io.File;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javax.imageio.ImageIO;

import com.googlecode.javacv.FFmpegFrameGrabber;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.cpp.avutil;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Segment;
import visualizer.GLVisualizer;

/**
 * A helper class that takes a video file and passes the video frames to the
 * owner visualizer instance. In the future, this class may do the segmentation
 * job as well and pass the segments to the visualizer.
 * 
 * @author Mehran Maghoumi
 * 
 */
public class FramePasser implements Runnable {
	/** OpenCV's frame grabber */
	private FFmpegFrameGrabber grabber = null;

	/** The visualizer instance that owns this FramePasser instance */
	private GLVisualizer owner = null;

	/** The worker thread that parses the video file */
	private Thread workerThread = null;

	/** The current frame */
	private int frameNumber = 1;

	/** The length of the video loaded by this instance in number of frames */
	private int lengthInFrames = 0;

	/** Worker thread's death signal */
	private boolean alive = true;

	private boolean paused = false;
	
	private boolean stepMode = false;
	

	public FramePasser(GLVisualizer owner, File videoFile) {
		avutil.av_log_set_level(avutil.AV_LOG_QUIET); // Shutup FFMpeg!
		this.owner = owner;
		this.grabber = new FFmpegFrameGrabber(videoFile);
		this.workerThread = new Thread(this);

		try {
			grabber.start();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		this.lengthInFrames = grabber.getLengthInFrames() - 1; // because of the bug in FFMpeg

		if (owner != null)
			this.owner.setVideoLength(this.lengthInFrames);
	}

	public FramePasser(File videoFile) {
		this(null, videoFile);
	}

	public void setOwner(GLVisualizer visualizer) {
		this.owner = visualizer;
		this.owner.setVideoLength(lengthInFrames);
	}

	/**
	 * @return The width of the video file
	 */
	public int getVideoWidth() {
		return this.grabber.getImageWidth();
	}

	/**
	 * @return The height of the video file
	 */
	public int getVideoHeight() {
		return this.grabber.getImageHeight();
	}

	/**
	 * Start the worker thread
	 */
	public void start() {
		this.alive = true;
		if (!this.workerThread.isAlive())
			this.workerThread.start();
	}

	/**
	 * Stop the worker thread
	 */
	public void stop() {
		this.alive = false;
	}

	/**
	 * Pauses the frame passer
	 */
	public void pause() {
		this.paused = true;
	}

	/**
	 * Resumes the frame passer
	 */
	public void resume() {
		this.paused = false;

		synchronized (this) {
			notify();
		}
	}

	/**
	 * Toggles the paused state and returns the current state.
	 * 
	 * @return The current paused state.
	 */
	public boolean togglePaused() {
		if (this.paused) {
			resume();
		}
		else
			pause();

		return this.paused;
	}

	/**
	 * @return Is the frame passer paused?
	 */
	public boolean isPaused() {
		return this.paused;
	}
	
	public boolean isStepMode() {
		return this.stepMode;
	}
	
	public void setStepMode(boolean stepMode) {
		this.stepMode = stepMode;
	}

	/**
	 * Set the position in the video file
	 * 
	 * @param frame
	 *            The frame number to jump to
	 */
	public void setVideoPosition(int frame) {
		if (this.frameNumber != frame) {
			synchronized (grabber) {
				this.frameNumber = frame;
				try {
					grabber.setFrameNumber(this.frameNumber);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
				}
			}
		}
	}

	@Override
	public void run() {
		try {
			// TODO start video here if not started (is it even necessary?)
			while (this.alive) {
				
				if (this.paused) {
					synchronized (this) {
						wait();
					}
				}
				
//				if (this.paused) // If this passer is paused, we shouldn't produce. Therefore, sleep tight :-]
//					synchronized (this) {
//						wait();
//					}

				//				if (this.frameNumber == this.lengthInFrames) { // Should the video be restarted?
				//					this.frameNumber = 1;
				//					this.grabber.restart();
				//				}

				ByteImage result = runOneFrame();
//				ByteImage result2 = runOneFrame();
				
//				ImageIO.write(result.getBufferedImage(), "png", new File("D:\\first-frame.png"));
//				ImageIO.write(result2.getBufferedImage(), "png", new File("D:\\second-frame.png"));
//				System.exit(0);
				
				this.owner.setCurrentFrame(++this.frameNumber);
//				this.owner.passNewFrame(result);
//				int x = 0, y = 0;
//				int width = 128, height = 128;
//				this.owner.passNewSegment(new Segment(result.getSubimage(prevFirstX, 0, width, height), prevFirstX, 0, width, height));
//				this.owner.passNewSegment(new Segment(result.getSubimage(prevSecondX, 263, width, height), prevSecondX, 263, width, height));
//				this.owner.passNewSegment(new Segment(result.getSubimage(484, 263, width, height), 484, 263, width, height));
				
				prevFirstX += 3;
				prevSecondX += 3;
//				ImageIO.write(result.getSubimage(26, 263, width, height).getBufferedImage(), "png", new File("D:\\firstt.png"));
//				ImageIO.write(result.getSubimage(484, 263, width, height).getBufferedImage(), "png", new File("D:\\secondd.png"));
//				System.exit(0);
				//				// Grab a frame and pass it to the owner
				//				IplImage grabbed = null;
				//				synchronized(this.grabber) {
				//					grabbed = this.grabber.grab();
				//				}
				//				this.owner.passNewFrame(new ByteImage(grabbed, this.grabber.getPixelFormat()));
			}

			// Here = not alive anymore
			this.grabber.stop();
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	int prevFirstX = 0;
	int prevSecondX = 26;

	public ByteImage runOneFrame() {
		try {
			if (this.frameNumber == this.lengthInFrames) { // Should the video be restarted?
				this.frameNumber = 1;
				this.grabber.restart();
				prevFirstX = 0;
				prevSecondX = 26;
			}

			IplImage grabbed = null;
			
			synchronized (this.grabber) {
//				grabber.setFrameNumber(1);
				grabbed = this.grabber.grab();
			}
			
			return new ByteImage(grabbed, this.grabber.getPixelFormat());
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		// should never happen 
		return null;
	}

}