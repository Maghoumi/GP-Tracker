package visualizer;

import invoker.Invoker;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.event.*;
import java.net.URL;
import java.nio.Buffer;
import java.nio.IntBuffer;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import javax.media.opengl.GL;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.border.EmptyBorder;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;

import net.miginfocom.swing.MigLayout;
import utils.ByteImage;
import utils.Classifier;
import utils.ClassifierSet;
import utils.ImageFilterProvider;
import utils.Segment;
import utils.SegmentEventListener;
import utils.SegmentEventProvider;
import utils.SegmentedVideoFrame;
import utils.ClassifierSet.ClassifierAllocationResult;
import utils.SuccessListener;
import utils.gui.CheckBoxList;
import utils.gui.ClassifierCheckBox;
import utils.gui.Slider;
import utils.opengl.OpenGLUtils;

import com.jogamp.opengl.util.Animator;
import com.sir_m2x.transscale.TransScale;

import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.ScrollPaneConstants;

/**
 * And finally... The JCuda/OpenGL visualizer. This class is basically a JFrame designed with WindowBuilder that has an OpenGL Canvas on it. This
 * class is able to run the visualizers evolved by ECJ on CUDA and paint their results using OpenGL. It also provides other functionalities which are
 * yet to be determined!
 * 
 * @author Mehran Maghoumi
 * 
 */
public class GLVisualizer extends JFrame implements GLEventListener, Visualizer, SegmentEventProvider {

	public static final URL ICON_PLAY = GLVisualizer.class.getResource("/icons/play-icon.png");
	public static final URL ICON_PAUSE = GLVisualizer.class.getResource("/icons/pause-icon.png");
	public static final URL ICON_FFW = GLVisualizer.class.getResource("/icons/ff-icon.png");
	public static final URL ICON_RW = GLVisualizer.class.getResource("/icons/rw-icon.png");
	public static final URL KERNEL_PATH = GLVisualizer.class.getResource("/cuda/kernels/visualizer/visualizer-kernel.cu");

	private static final int REPORT_FREQUENCY = 50;

	/** The kernel wrapper object to use for invoking CUDA */
	private GLVisualizerKernel kernel;

	/** The width of the image (frame) to be displayed */
	private int imageWidth;

	/** The height of the image (frame) to be displayed */
	private int imageHeight;

	/** The shared OpenGL/CUDA buffer */
	private int glBuffer;

	/** OpenGL texture handle */
	private int glTexture;

	/** OpenGL animator that constantly calls the Display() function */
	private Animator animator;

	/** A flag to indicate that this visualizer is loaded and ready for operation */
	private boolean isReady = false;

	/** The frame and individual queue capacity */
	private final int QUEUE_CAPACITY = 1;

	/** The frame queue which holds the frames and the discovered segments that we want to display using OpenGL */
	private BlockingQueue<SegmentedVideoFrame> segmentedFrameQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

	/**
	 * A set of the evolved classifiers. When a new classifier is passed to this object, that classifier is stored in this set. Also, when the CUDA
	 * kernel needs to be called, the classifiers in this set are passed to the kernel function as GP individuals
	 */
	private ClassifierSet classifiers = new ClassifierSet();
	
	/** A flag indicating whether the newest classifier meets the requirements and is good enough */
	protected volatile boolean eagerTerminateSatisfied = false;
	
	/** Synchronization for the passNewClassifier function */
	protected Object passedClassifierMutex = new Object();

	/** The associated invoker reference */
	private Invoker invoker;

	/** A flag indicating whether filtering is enabled or not */
	private boolean filterEnabled = true;
	
	/** The set of all SuccessListeners */
	private Set<SuccessListener> successListeners = new HashSet<>();
	
	/** The set of all SegmentEventListeners */
	private Set<SegmentEventListener> segmentListeners = new HashSet<>();
	
	/** A set of all permanent orphans */
	private Set<String> permanentOrphans = new HashSet<>();
	
	/** Keeps track of the number of segments in the current video frame */
	private volatile int currentSegmentCount = 2;//FIXME constant!
	
	/** Keeps track of the number of segments in the previous video frame */
	private volatile int previousSegmentCount = 2;//FIXME constant!

	/** FPS helper variable */
	private long prevTime = -1;
	/** FPS helper variable */
	private int step = 0;
	/** The last calculated framerate */
	private double framerate = 0;

	private JPanel contentPane;
	private JPanel pnlContent;
	private JPanel pnlOpenGL;
	private GLCanvas glComponent;
	private JPanel panel;
	private JButton btnResetSize;
	private JPanel pnlRight;
	private CheckBoxList checkBoxList;
	private JButton btnSeedRetrain;
	private JPanel pnlCenter;
	private JPanel pnlVideoControls;
	private Slider sliderVidPosition;
	private JButton btnRew;
	private JButton btnPlayPause;
	private JButton btnFF;
	private JCheckBox chckbxShowConflicts;
	private JButton btnRetrain;
	private JCheckBox chckbxThreshold;
	private JSpinner spnThreshold;
	private JPanel panel_3;
	private JLabel lblNewLabel_1;
	private JPanel panel_4;
	private JLabel lblOpacity;
	private JSpinner spnOpacity;
	private JToggleButton tglbtnGpSystem;
	private JButton btnSelectAll;
	private JButton btnSelectNone;
	private JButton btnDel;

	/**
	 * Initializes the visualizer with the given image width and height. Use this constructor if you want to dynamically pass data to the visualizer.
	 * 
	 * @param imageWidth
	 *            The width of the image to be visualized
	 * @param imageHeight
	 *            The height of the image to be visualized
	 * 
	 */
	public GLVisualizer(int imageWidth, int imageHeight) {
		this.imageWidth = imageWidth;
		this.imageHeight = imageHeight;

		setup();
	}

	public GLVisualizer(Invoker invoker) {
		this(invoker.getFrameWidth(), invoker.getFrameHeight());
		this.invoker = invoker;
	}

	/**
	 * Fake constructor! Only here for the purposes of Eclipse's WindowBuilder plugin.
	 * 
	 * @wbp.parser.constructor
	 */
	private GLVisualizer() {
		setPreferredSize(new Dimension(500, 550));

		//		setBounds(100, 100, 467, 338);
		setup();
	}

	/**
	 * Sets up the UI elements. Should be used by WindowBuilder. Try not to touch!
	 */
	private void setupUIElements() {
		setTitle("JCuda/JOGL Visualizer");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(new BorderLayout(0, 0));
		this.panel = new JPanel();
		this.contentPane.add(this.panel, BorderLayout.CENTER);
		this.panel.setLayout(new BorderLayout(5, 5));

		pnlContent = new JPanel();
		this.pnlContent.setMinimumSize(new Dimension(200, 200));
		this.panel.add(this.pnlContent, BorderLayout.CENTER);
		this.pnlContent.setLayout(new BorderLayout(0, 0));

		pnlRight = new JPanel();
		pnlRight.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "Options", TitledBorder.LEADING, TitledBorder.TOP,
				null, null));
		pnlContent.add(pnlRight, BorderLayout.EAST);
		pnlRight.setLayout(new MigLayout("", "[grow]", "[150px:n,fill][][center][][][][][][][][][][]"));

		checkBoxList = new CheckBoxList();
		checkBoxList.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		checkBoxList.setPreferredSize(new Dimension(130, 130));
		pnlRight.add(checkBoxList, "cell 0 0,grow");

		btnSeedRetrain = new JButton("Seed Retrain");
		btnSeedRetrain.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Object selected = checkBoxList.getSelectedValue();
				if (selected == null)
					return;

				Classifier classifier = ((ClassifierCheckBox) selected).getBoundedClassifier();
				invoker.retrain(classifier, true);
			}
		});
		
		btnSelectAll = new JButton("All");
		btnSelectAll.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				checkBoxList.selectAll();
			}
		});
		pnlRight.add(btnSelectAll, "flowx,cell 0 1,growx");

		chckbxShowConflicts = new JCheckBox("Show conflicts");
		pnlRight.add(chckbxShowConflicts, "cell 0 2,growx");
		
		chckbxEagerTermination = new JCheckBox("Eager termination");
		chckbxEagerTermination.setSelected(true);
		pnlRight.add(chckbxEagerTermination, "cell 0 3");

		chckbxThreshold = new JCheckBox("Do thresholding");
		chckbxThreshold.setEnabled(false);
		chckbxThreshold.setSelected(true);
		pnlRight.add(chckbxThreshold, "cell 0 4,growx");

		panel_3 = new JPanel();
		pnlRight.add(panel_3, "cell 0 5,growx");
		panel_3.setLayout(new BorderLayout(0, 0));

		lblNewLabel_1 = new JLabel(" Threshold:");
		panel_3.add(lblNewLabel_1, BorderLayout.WEST);

		spnThreshold = new JSpinner();
		spnThreshold.setPreferredSize(new Dimension(35, 20));
		panel_3.add(spnThreshold, BorderLayout.EAST);
		spnThreshold.setModel(new SpinnerNumberModel(50, 0, 100, 1));

		this.panel_4 = new JPanel();
		this.pnlRight.add(this.panel_4, "cell 0 6,growx");
		this.panel_4.setLayout(new BorderLayout(0, 0));

		this.lblOpacity = new JLabel(" Opacity:");
		this.panel_4.add(this.lblOpacity, BorderLayout.WEST);

		this.spnOpacity = new JSpinner();
		this.spnOpacity.setModel(new SpinnerNumberModel(50, 0, 100, 1));
		this.spnOpacity.setPreferredSize(new Dimension(35, 20));
		this.panel_4.add(this.spnOpacity, BorderLayout.EAST);
		pnlRight.add(btnSeedRetrain, "cell 0 8,growx");

		btnRetrain = new JButton("Retrain");
		btnRetrain.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Object selected = checkBoxList.getSelectedValue();
				if (selected == null)
					return;

				Classifier classifier = ((ClassifierCheckBox) selected).getBoundedClassifier();
				invoker.retrain(classifier, false);
			}
		});
		pnlRight.add(btnRetrain, "cell 0 9,growx");

		btnResetSize = new JButton("Reset Canvas");
		this.pnlRight.add(this.btnResetSize, "cell 0 10,growx");

		tglbtnGpSystem = new JToggleButton("GP [Enabled]", true);
		tglbtnGpSystem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (tglbtnGpSystem.isSelected())
					tglbtnGpSystem.setText("GP [Enabled]");
				else
					tglbtnGpSystem.setText("GP [Disabled]");

				invoker.setGPStatus(tglbtnGpSystem.isSelected());
			}
		});
		pnlRight.add(tglbtnGpSystem, "cell 0 12,growx");
		
		btnSelectNone = new JButton("None");
		btnSelectNone.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				checkBoxList.selectNone();
			}
		});
		btnSelectNone.setMinimumSize(new Dimension(0, 0));
		pnlRight.add(btnSelectNone, "cell 0 1,growx,aligny center");
		
		btnDel = new JButton("Del");
		btnDel.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Object selected = checkBoxList.getSelectedValue();
				if (selected == null)
					return;

				Classifier classifier = ((ClassifierCheckBox) selected).getBoundedClassifier();
				removeClassifier(classifier);
			}
		});
		btnDel.setPreferredSize(new Dimension(40, 23));
		pnlRight.add(btnDel, "cell 0 1");
		btnResetSize.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
//				glComponent.setPreferredSize(new Dimension(imageWidth, imageHeight));
				glComponent.setPreferredSize(new Dimension(854, 800));
				pack();
			}
		});
	}
	
	BlockingQueue<Boolean> eagerQueue = new ArrayBlockingQueue<>(1);
	volatile Thread waiting = null;
	private JCheckBox chckbxEagerTermination;

	/**
	 * Used to pass a new individual to the visualizer. Usually, when a fitter individual is found in the run, this fncDescribe should be called. Note
	 * that the calling thread will be blocked if the individual queue of this instance is full.
	 * 
	 * @param classifier
	 *            The new classifier that is found during the GP run
	 */
	public boolean passNewClassifier(Classifier classifier) {
		synchronized (this.classifiers) {
			waiting = Thread.currentThread();
			this.classifiers.update(classifier);
			this.checkBoxList.addItem(classifier);
		}
		
		boolean result = false;
		
		try {
			result = eagerQueue.take();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		waiting = null;
		
		if (chckbxEagerTermination.isSelected())
			return !result;
		else
			return true;
	}

	/**
	 * Removes the classifiers from the list of the evolved classifiers
	 * @param classifiers
	 */
	public void removeClassifier(Collection<Classifier> classifiers) {
		for (Classifier c : classifiers)
				removeClassifier(c);
	}
	
	/**
	 * Removes the passed classifier from the list of evolved classifiers
	 * and destroys it by means of releasing its color
	 * @param classifier
	 */
	protected void removeClassifier(Classifier classifier) {
		synchronized (this.classifiers) {
			this.checkBoxList.removeItem(classifier);
			this.classifiers.remove(classifier);
			classifier.destroy();	// release the color
		}
	}

	/**
	 * Used to pass a new video frame (image) to the visualizer. Usually should be called after grabbing a frame of the video. Note that the calling
	 * thread will be blocked if the frame queue of the visualizer is full
	 * 
	 * @param frame
	 *            The new frame that is obtained from the video/image
	 */
	public void passNewFrame(SegmentedVideoFrame frame, int currentPosition) {
		try {
			segmentedFrameQueue.put(frame);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
		setCurrentFrame(currentPosition);
	}

	/**
	 * Initializes the OpenGL canvas and window. Usually this method must be called after the constructor. Note that the calling thread will be
	 * blocked until the canvas is ready and OpenGL thread has done its initialization magic.
	 */
	private void setup() {

		setupUIElements();

		GLProfile profile = GLProfile.get(GLProfile.GL2);
		final GLCapabilities capabilities = new GLCapabilities(profile);
		KeyboardControl keyListener = new KeyboardControl();

		// Create the main frame 
		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				animator.stop();
			}
		});

		pnlCenter = new JPanel();
		pnlContent.add(pnlCenter, BorderLayout.CENTER);
		pnlCenter.setLayout(new BorderLayout(0, 0));
		this.pnlOpenGL = new JPanel();
		pnlCenter.add(pnlOpenGL);
		this.pnlOpenGL.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "OpenGL", TitledBorder.LEADING,
				TitledBorder.TOP, null, null));

		// Initialize the GL component and the animator
		glComponent = new GLCanvas(capabilities);
		glComponent.setFocusable(true);
		glComponent.addGLEventListener(this);
		glComponent.addKeyListener(keyListener);
		this.pnlOpenGL.setLayout(new BorderLayout(0, 0));

		glComponent.setPreferredSize(new Dimension(imageWidth, imageHeight));

		this.pnlOpenGL.add(glComponent);

		animator = new Animator(glComponent);

		pnlVideoControls = new JPanel();
		this.pnlContent.add(this.pnlVideoControls, BorderLayout.SOUTH);
		pnlVideoControls.setLayout(new MigLayout("", "[][][][grow]", "[]"));

		btnRew = new JButton("");
		btnRew.setIcon(new ImageIcon(ICON_RW));
		pnlVideoControls.add(btnRew, "cell 0 0");

		btnPlayPause = new JButton("");
		this.btnPlayPause.setIcon(new ImageIcon(ICON_PAUSE));
		btnPlayPause.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (GLVisualizer.this.invoker.togglePaused())
					btnPlayPause.setIcon(new ImageIcon(ICON_PLAY));
				else
					btnPlayPause.setIcon(new ImageIcon(ICON_PAUSE));
			}
		});
		btnPlayPause.setFont(new Font("Tahoma", Font.PLAIN, 11));
		pnlVideoControls.add(btnPlayPause, "cell 1 0");

		btnFF = new JButton();
		btnFF.setIcon(new ImageIcon(ICON_FFW));
		this.btnFF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				throw new RuntimeException("Not implemented yet");
				//												GLVisualizer.this.videoFeeder.pause();
				//												try {
				//													Thread.sleep(1000);
				//												} catch (InterruptedException e1) {
				//													e1.printStackTrace();
				//												}
				//												
				//												GLVisualizer.this.segmentedFrameQueue.clear();
				//												GLVisualizer.this.videoFeeder.getAndPassNextFrame();
			}
		});
		pnlVideoControls.add(btnFF, "cell 2 0");
		sliderVidPosition = new Slider();
		this.sliderVidPosition.addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				//												GLVisualizer.this.videoFeeder.pause();
				throw new RuntimeException("Not implemented yet");
			}

			@Override
			public void mouseReleased(MouseEvent e) {
				//												GLVisualizer.this.segmentedFrameQueue.clear();
				//												GLVisualizer.this.videoFeeder.setCurrentPosition(sliderVidPosition.getValue());
				//												GLVisualizer.this.videoFeeder.resume();
				throw new RuntimeException("Not implemented yet");
			}
		});
		this.sliderVidPosition.setValue(0);
		this.sliderVidPosition.setMajorTickSpacing(1);
		this.sliderVidPosition.setMinorTickSpacing(1);
		sliderVidPosition.setMinimum(1);
		pnlVideoControls.add(sliderVidPosition, "cell 3 0,growx");
		animator.setUpdateFPSFrames(30, null); // animator will use 30 FPS
		animator.start();
		
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
	    GraphicsDevice[] gd = ge.getScreenDevices();
	    
	    // Force it to appear on the other screen! FOR THESIS FINALIZATION ONLY
//	    if (gd.length != 1) {
//	    	setLocation(gd[1].getDefaultConfiguration().getBounds().x, getY());	    	
//	    }

		this.pack();
		this.setVisible(true);
		// Block the calling thread until OpenGL is ready and fully loaded
		waitReady();
	}

	@Override
	public void init(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		// set erase color to white
		gl.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		gl.glDisable(GL.GL_DEPTH_TEST);

		// Initialize CUDA
		this.kernel = new GLVisualizerKernel();

		initBuffers(drawable);
		initTexture(drawable);
		this.isReady = true; // Indicate that I am ready an loaded! Gimme data!
	}

	public void initBuffers(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();

		int[] buffer = new int[1];

		// Generate buffer
		gl.glGenBuffers(1, IntBuffer.wrap(buffer));
		glBuffer = buffer[0];

		/*
		 * Bind the generated buffer. Note that I am using the pixel unpack buffer here. I think the reason is that I want to bind the data to a
		 * texture later. Per OpenGL's documentation, glTexSubImage2D will use pixel unpack buffer data if a buffer is bound to pixel unpack buffer.
		 */
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);
		// Specify the size of the buffer (no data is pre-loaded in this buffer)
		gl.glBufferData(GL2.GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, (Buffer) null, GL2.GL_DYNAMIC_COPY);
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0); // Unbind pixel unpack buffer

		this.kernel.registerGLBuffer(glBuffer);
	}

	public void initTexture(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		int[] texture = new int[1];

		gl.glGenTextures(1, IntBuffer.wrap(texture));
		glTexture = texture[0];

		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);

		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_LINEAR);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_LINEAR);

		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_RGBA8, imageWidth, imageHeight, 0, GL2.GL_BGRA, GL2.GL_UNSIGNED_BYTE, (Buffer) null);

		gl.glBindTexture(GL2.GL_TEXTURE_2D, 0); // I put it here!

	}

	@Override
	public void display(GLAutoDrawable drawable) {

		// Will invoke CUDA if necessary
		if (this.invoker == null)
			return;

		if (filterEnabled)
			runCuda(drawable);
		else
			copyCat(drawable);

		GL2 gl = drawable.getGL().getGL2();

		// Bind current buffer to the pixel unpack buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);

		// Bind texture
		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);

		// Ask OpenGL to use the data which exist in the pixel unpack buffer as the texture
		/**
		 * Also note that RGBA was specified earlier, however right now, I have asked OpenGL to do the pixel format conversion
		 */
		gl.glTexSubImage2D(GL2.GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL2.GL_ABGR_EXT, GL2.GL_UNSIGNED_BYTE, 0); //ZERO! NOT NULL! :-) because I want the content of the PIXEL_UNPACK_BUFFER to be used

		// Unbind pixel unpack buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

		gl.glEnable(GL2.GL_TEXTURE_2D);
		gl.glDisable(GL2.GL_DEPTH_TEST);
		gl.glDisable(GL2.GL_LIGHTING);
		gl.glTexEnvf(GL2.GL_TEXTURE_ENV, GL2.GL_TEXTURE_ENV_MODE, GL2.GL_REPLACE);

		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glPushMatrix();
		gl.glLoadIdentity();
		gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glBegin(GL2.GL_QUADS);
		gl.glTexCoord2f(0.0f, 1.0f);
		gl.glVertex2f(-1.0f, -1.0f);

		gl.glTexCoord2f(1.0f, 1.0f);
		gl.glVertex2f(1.0f, -1.0f);

		gl.glTexCoord2f(1.0f, 0.0f);
		gl.glVertex2f(1.0f, 1.0f);

		gl.glTexCoord2f(0.0f, 0.0f);
		gl.glVertex2f(-1.0f, 1.0f);
		gl.glEnd();

		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glPopMatrix();

		gl.glDisable(GL2.GL_TEXTURE_2D);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, 0); // I put it here

		calculateFramerate();

	}

	/**
	 * Helper function that invokes CUDA if necessary. Will also determine if classifiers
	 * have issues and need to be retrained/recreated
	 * 
	 * @param drawable
	 *            OpenGL drawable
	 */
	public void runCuda(GLAutoDrawable drawable) {
		
		/**
		 * When CUDA is to be called, the classifiers must be locked so that
		 * CUDA can finish processing and then GP can interfere
		 * Otherwise, in the middle of processing, a classifier may be updated
		 * or added to the system and we may not be able to see that it has actually
		 * changed
		 */
		boolean cycleSuccessful = true;	// Flag indicating the success of this session
		
		synchronized (this.classifiers) {
			// This describe kernel only needs to be invoked if:
			// 		1) There is a new frame in the frame queue 
			//		2) There is a new individual in the individual queue **AND** we
			//		   have had at least one frame in our frame queue
			//
			// If neither of these conditions are true, the OpenGL buffer
			// should not be touched and CUDA should not be invoked

			// Check my queues and see what I should do
			SegmentedVideoFrame frame = segmentedFrameQueue.poll();

			// Either way we need to copy the frame in the queue to the OpenGL buffer
			copyCat(drawable);

			if (frame == null)
				return;
			
			// Invoke kernel and determine orphans
			List<Segment> orphans = invokeKernel(drawable, frame);
			
//			if (!invoker.isQueueEmpty())
//				return;
			eagerTerminateSatisfied = true;
			
			// Determine permanent orphans
			determinePermanentOrphans(frame, orphans);
			
			// Resolve training issues and remove garbage/wrong classifiers
			cycleSuccessful = !resolveIssues(frame);
			
			currentSegmentCount = frame.size();
			if (previousSegmentCount < currentSegmentCount) {
				// Means a new segment has been added to the system
				/*
				 * Now we need to invoke CUDA, resolve training issues and
				 * again invoke CUDA to determine how many orphans we will
				 * be left with now that a new segment has been added
				 */
				orphans = invokeKernel(drawable, frame);
				cycleSuccessful = !resolveIssues(frame);
				determinePermanentOrphans(frame, orphans);
				orphans = invokeKernel(drawable, frame);
				
				notifySegmentAdded(frame.size(), orphans.size(), permanentOrphans.size());
			}
			
			previousSegmentCount = currentSegmentCount;
			
			if (!orphans.isEmpty() && invoker.isQueueEmpty()) {
				Segment toBeTrained = null;
				for (Segment s : orphans) {
					if (!s.isPermanentOrphan()) {
						toBeTrained = s;
						cycleSuccessful = false;
						break;
					}
				}
				
				invoker.evolveClassifier(frame, toBeTrained);
			}
			
//			if (!invoker.isQueueEmpty())
//				return;
			
			if (this.classifiers.isEmpty() || !frame.hasSegments())
				cycleSuccessful = false;
			
			if (!invoker.isQueueEmpty())
				cycleSuccessful = false;
			
			if(classifiers.isEmpty())
				eagerTerminateSatisfied = false;
						
			if (waiting != null)
				eagerQueue.offer(eagerTerminateSatisfied);
			
				
			if (cycleSuccessful)
				notifySuccess();

		}/* end-synchronized*/

	}
	
	/**
	 * Invokes the CUDA kernel on the passed frame, draws the overlays and returns the
	 * list of orphan textures in this frame.
	 * 
	 * @param drawable	JOGL drawable
	 * @param frame	The frame to run CUDA for
	 * @return	a list of all orphan textures in this frame
	 */
	private List<Segment> invokeKernel(GLAutoDrawable drawable, SegmentedVideoFrame frame) {
		// Create a list to maintain orphans
		// NOTE: Must be synchronized for a dual-card setup
		List<Segment> orphans = Collections.synchronizedList(new ArrayList<Segment>());
		classifiers.resetClaims();	// Reset the claimes since we are going to run them all
		ClassifierAllocationResult pointerToAll = this.classifiers.getPointerToAll();
		
		if (pointerToAll == null) {
			for (Segment s : frame) {
				orphans.add(s);
			}
			
			return orphans;
		}
		
		frame.shuffle();	// Shuffle the frames so that each time we will process a different segment
		
		// Delegate the processing to a single thread
		int gpuCount = TransScale.getInstance().getNumberOfDevices();
		int portion = frame.size() / gpuCount;
		int offset = 0;
		Thread[] threads = new Thread[gpuCount];
		
		for (int i = 0 ; i < gpuCount ; i++) {
			Describer d = new Describer();
			d.orphans = orphans;
			d.drawable = drawable;
			
			// Adjust the number of segments for the last thread
			if (i == gpuCount - 1) {
				portion = frame.size() - i * portion;
				d.pointerToAll = pointerToAll;	// optimization! use the already existing pointerToAll for one of them
			}
			else {
				d.pointerToAll = pointerToAll == null ? null : (ClassifierAllocationResult) pointerToAll.clone();
			}
			
			for (int j = offset ; j < offset + portion ; j++) {
				d.mySegments.add(frame.get(j));
			}
			
			offset += portion;
			
			threads[i] = new Thread(d);
			threads[i].setName("Visualizer spawned #" + i);
			threads[i].start();			
		}
		
		for (int i = 0 ; i < gpuCount ; i++) {
			try {
				threads[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		drawOverlay(drawable);
		
		return orphans;
	}
	
	/**
	 * Called by the invokeKernel method after each kernel invocation in order to
	 * draw the color overlays based on the results of the kernel execution.
	 * 
	 * @param drawable	JOGL drawable
	 */
	private void drawOverlay(GLAutoDrawable drawable) {
		// The mapping of segment => color
		Map<Rectangle, Color> mapping = new HashMap<>();
		float opacity = Float.valueOf(spnOpacity.getValue().toString()).floatValue() / 100f;
		
		for (Classifier c : this.classifiers) {
			if (!c.isEnabled())
				continue;
			
			for (Segment claimed : c.getClaims()) {
				Rectangle bounds = claimed.getBounds();
				Color color = c.getColor();
				
				if (!mapping.containsKey(bounds)) {		// If this pair does not exist add it to the list of the overlays we have to do
					mapping.put(bounds, color);
				}
				else {
					mapping.put(bounds, Color.RED);		// Colorize this overlay by red indicating a problem.
				}
			}	// end for (claimed segments)
		} // end for (classifiers)
		
		// Draw the overlays based on the mapping
		for (Entry<Rectangle, Color> entry : mapping.entrySet()) {
			OpenGLUtils.drawRegionOverlay(drawable, glBuffer, entry.getValue(), opacity, imageWidth, imageHeight, entry.getKey());			
		}
	}

	/**
	 * Removes the unnecessary classifiers from the system and detects
	 * and resolves the training issues.
	 * @return	True if an issue was detected, false otherwise
	 */
	private boolean resolveIssues(SegmentedVideoFrame frame) {
		boolean issueDetected = false;
		
//		for (Classifier c : this.classifiers) {
//			if (c.isBeingProcessed())
//				return true;
//		}
		
		List<Classifier> toBeDestroyed = new ArrayList<>();
		
		// Remove garbage classifiers
		for (Classifier c : classifiers) {
			if (c.getClaimsCount() == 0 /*&& !c.isBeingProcessed()*/ && chckbxThreshold.isSelected() /*&& invoker.isQueueEmpty()*/) {	// If this classifier has not claimed anything and is not currently being processed, it's garbage and must be deleted! :-)
				toBeDestroyed.add(c);
				issueDetected = true;
			}				
			else if (c.getClaimsCount() > 1 /*&& !c.isBeingProcessed()*/ && chckbxThreshold.isSelected() /*&& invoker.isQueueEmpty()*/) {	// If this classifier has not claimed anything and is not currently being processed, it's garbage and must be deleted! :-)
				toBeDestroyed.add(c);
				issueDetected = true;
			}
		}
		
		// Destroy classifiers that have claimed the same segments
		Map<Segment,List<Classifier>> singleClaimers = new HashMap<>();	// We do not want to over-destroy! Leave one of the single claimers and destroye the rest :-)
		
		for (Segment s : frame) {
			List<Classifier> listOfSingles = new ArrayList<>();
			
			if (s.getClaimersCount() == 1) {
				singleClaimers.put(s, listOfSingles);
				continue;
			}
			
			for (Classifier c : s.getClaimers()) {
				if (c.getClaimsCount() == 1  && c.getClaims().contains(s) /*&& !c.isBeingProcessed()*/ && chckbxThreshold.isSelected() /*&& invoker.isQueueEmpty()*/) {
					listOfSingles.add(c);
					issueDetected = true;
				}
				
				singleClaimers.put(s, listOfSingles);			
			}
		}
		
		// Remove older ones from single claimers
		for (Entry<Segment, List<Classifier>> entry : singleClaimers.entrySet()) {
			List<Classifier> currentSingleClaimers = entry.getValue();
			
			if (currentSingleClaimers.size() > 0) {
				// Remove the older classifier so that we can safely remove the rest
				long minTime = Long.MAX_VALUE;
				int index = -1;
				
				for (int i = 0 ; i < currentSingleClaimers.size() ; i++) {
					Classifier c = currentSingleClaimers.get(i);
					if (c.timestamp < minTime) {
						minTime = c.timestamp;
						index = i;
					}
				}
				
				currentSingleClaimers.remove(index);
				
				// Mark all the remaining classifiers to be destroyed
				for (Classifier c : currentSingleClaimers) {
					toBeDestroyed.add(c);
				}
			}
		}
		
		eagerTerminateSatisfied = !issueDetected;

		// Destroy the suckers
		if (invoker.isQueueEmpty())
			removeClassifier(toBeDestroyed);
		
		return issueDetected;
	}
	
	/**
	 * Determines the permanent orphan textures and removes them from the list of orphans
	 * 
	 * @param frame
	 * @param orphans
	 */
	private void determinePermanentOrphans(SegmentedVideoFrame frame, List<Segment> orphans) {
		synchronized (this.permanentOrphans) {
			for (Segment s : frame)
				if (invoker.hasChoked(s)) {
					s.setPermanentOrphan(true);
					permanentOrphans.add(s.toString());
				}
			
			Iterator<Segment> iter = orphans.iterator();
			while (iter.hasNext()) {
				Segment item = iter.next();
				if (item.isPermanentOrphan())
					iter.remove();
			}
		}
	}

	/**
	 * Bypasses CUDA and will directly copy the frame in the frame queue to the OpenGL buffer.
	 * 
	 * @param drawable
	 *            OpenGL drawable
	 */
	private void copyCat(GLAutoDrawable drawable) {
		// Grab the next frame from the frame queue
		ByteImage frame = null;

		try {
			frame = segmentedFrameQueue.take().getFrame();
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}

		OpenGLUtils.copyBuffer(frame, glBuffer, drawable);
	}

	private void calculateFramerate() {
		step++;
		long currentTime = System.nanoTime();
		if (prevTime == -1) {
			prevTime = currentTime;
		}
		long diff = currentTime - prevTime;
		if (diff > 1e9) {
			double fps = (diff / 1e9) * step;
			String t = String.format("%.2f", fps);
			((TitledBorder) this.pnlOpenGL.getBorder()).setTitle("OpenGL - " + t + "fps");
			this.pnlOpenGL.repaint();

			prevTime = currentTime;
			step = 0;
			this.framerate = fps;
		}
	}

	@Override
	public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
		GL2 gl = drawable.getGL().getGL2();
		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();
	}

	@Override
	public void dispose(GLAutoDrawable drawable) {/* Do nothing */
	}

	/**
	 * Blocks the calling thread until this instance of the visualizer is loaded and ready to accept data.
	 */
	public void waitReady() {
		System.out.println(System.lineSeparator() + "Waiting for the visualizer to get ready...");

		while (!isReady)
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
	}

	/**
	 * A key adapter for JOGL key even functionalities
	 * 
	 * @author Mehran Maghoumi
	 */
	public class KeyboardControl extends KeyAdapter {
		public void keyTyped(KeyEvent e) {
			char c = e.getKeyChar();
			if (c == 'f') {
				filterEnabled = !filterEnabled;
			}
			if (c == ' ') {
				invoker.togglePaused();
			}
		}
	}

	/**
	 * Set the length of the video that this visualizer shows. This method will set the maximum value for the slider that shows the video position.
	 * 
	 * @param lengthInFrames
	 *            The video length in frames
	 */
	public void setVideoLength(int lengthInFrames) {
		this.sliderVidPosition.setMaximum(lengthInFrames);
	}

	/**
	 * Set the current position in the video. This will update the slider
	 * 
	 * @param frameNumber
	 */
	public synchronized void setCurrentFrame(Integer frameNumber) {
		this.sliderVidPosition.setValue(frameNumber);
	}

	@Override
	public void setVideoDimensions(int width, int height) {
		this.imageWidth = width;
		this.imageHeight = height;
		setPreferredSize(new Dimension(width, height));
		pack();
	}

	@Override
	public ImageFilterProvider getImageFilterProvider() {
		return this.kernel;
	}

	@Override
	public double getFramerate() {
		return this.framerate;
	}

	@Override
	public void addSuccessListener(SuccessListener listener) {
		this.successListeners.add(listener);
	}

	@Override
	public void removeSuccessListener(SuccessListener listener) {
		this.successListeners.remove(listener);
	}

	@Override
	public void notifySuccess() {
		for (SuccessListener listener : this.successListeners)
			listener.notifySuccess(this);
	}

	@Override
	public void notifyFailure(String reason) {
		// Do nothing
	}
	
	/**
	 * @return	The set of all the IDs of permanent orphan textures 
	 */
	public Set<String> getPermanentOrphans() {
		Set<String> result = new HashSet<>();
		
		synchronized (this.permanentOrphans) {
			for (String s : permanentOrphans)
				result.add("" + s);
		}		
		
		return result;
	}

	@Override
	public void addSegmentEventListener(SegmentEventListener listener) {
		this.segmentListeners.add(listener);
	}

	@Override
	public void removeSegmentEventListener(SegmentEventListener listener) {
		this.segmentListeners.remove(listener);
	}

	@Override
	public void notifySegmentAdded(int segmentCount, int orphansCount, int permanentOrphansCount) {
		for (SegmentEventListener listener : this.segmentListeners) {
			listener.segmentAdded(segmentCount, orphansCount, permanentOrphansCount);
		}
	}
	
	/**
	 * Helper class for multithread functionality. The threads backed by this Runnable
	 * will work on a specific portion of the segments. Each thread will run one segment on 1 card 
	 * and determine its claimers
	 * 
	 * @author Mehran Maghoumi
	 *
	 */
	private class Describer implements Runnable {
		
		/** The segments that this thread must process */
		public List<Segment> mySegments = new ArrayList<>();
		
		/** The list of orphans detected by this thread */
		public List<Segment> orphans;
		
		public ClassifierAllocationResult pointerToAll;
		
		public GLAutoDrawable drawable;

		@Override
		public void run() {
			for (Segment s : mySegments) {
				float threshold = Float.valueOf(spnThreshold.getValue().toString()).floatValue() / 100f;
				float opacity = Float.valueOf(spnOpacity.getValue().toString()).floatValue() / 100f;

				// If there are no classifiers, then there are no claimers for this texture!
				int claimers = (pointerToAll == null) ? 0 : kernel.call(invoker, drawable, pointerToAll, s,
						chckbxThreshold.isSelected(), threshold, opacity,
						chckbxShowConflicts.isSelected(), imageWidth, imageHeight);
				
				if (claimers == 0) {
					s.setOrphan(true);
					orphans.add(s);
				}
			}
		}
		
	}
}
