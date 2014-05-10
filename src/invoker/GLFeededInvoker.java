package invoker;

import java.lang.Thread.UncaughtExceptionHandler;

import javax.swing.UIManager;

import feeder.GLFeeder;
import utils.SuccessListener;
import visualizer.GLVisualizer;
import visualizer.Visualizer;

public class GLFeededInvoker extends Invoker{

	public GLFeededInvoker(String sessionPrefix) {
		super(new String[] { "-file", "bin/gp/object-tracker.params" }, sessionPrefix);
		this.feeder = new GLFeeder();
		this.visualizer = new GLVisualizer(this);
		visualizer.addSuccessListener(this);
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
		
		String prefix;
		if (args.length == 0)
			prefix = null;
		else
			prefix = args[0];
		
		new GLFeededInvoker(prefix).start();
	}

	@Override
	public void notifySuccess(Visualizer visualizer) {
		
	}

	@Override
	public void notifyFailure(String reason) {
		
	}

}
