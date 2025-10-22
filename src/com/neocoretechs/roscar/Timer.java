package com.neocoretechs.roscar;

import java.util.concurrent.TimeUnit;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public interface Timer extends AutoCloseable {
	static final Log log = LogFactory.getLog(Timer.class);
	@Override
	void close(); // no Exception

	static Timer log(String label) {
		return log(label, TimeUnit.MILLISECONDS);
	}

	static Timer log(String label, TimeUnit timeUnit) {
		return new Timer() {
			final long startNanos = System.nanoTime();

			@Override
			public void close() {
				long elapsedNanos = System.nanoTime() - startNanos;
				log.info(label + ": "
						+ timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
						+ timeUnit.toChronoUnit().name().toLowerCase());
			}
		};
	}
}

