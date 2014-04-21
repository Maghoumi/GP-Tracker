package invoker;

import java.lang.Thread.UncaughtExceptionHandler;

import javax.swing.UIManager;

import feeder.GLFeeder;
import visualizer.GLVisualizer;

public class GLFeededInvoker extends Invoker {

	public GLFeededInvoker() {
		super(new String[] { "-file", "bin/gp/object-tracker.params" });
		this.feeder = new GLFeeder();
		this.visualizer = new GLVisualizer(this);
	}
	
	public static void main(String[] args) throws Throwable {
		UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		Thread.setDefaultUncaughtExceptionHandler(new UncaughtExceptionHandler() {
			
			@Override
			public void uncaughtException(Thread arg0, Throwable arg1) {
				arg1.printStackTrace();
				System.exit(1); 
			}
		});
		new GLFeededInvoker().start();
	}

}
