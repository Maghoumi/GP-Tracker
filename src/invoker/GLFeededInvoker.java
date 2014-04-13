package invoker;

import javax.swing.UIManager;

import video_interop.OpenGLFeeder;
import visualizer.OpenGLVisualizer;

public class GLFeededInvoker extends Invoker {

	public GLFeededInvoker() {
		super(new String[] { "-file", "bin/m2xfilter/m2xfilter.params" });
		this.feeder = new OpenGLFeeder();
		this.visualizer = new OpenGLVisualizer(this);
	}
	
	public static void main(String[] args) throws Throwable {
		UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		new GLFeededInvoker().start();
	}

}
