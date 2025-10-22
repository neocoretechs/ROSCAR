package com.neocoretechs.roscar;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
		float temperature, float topp, long seed, int maxTokens, boolean preload, boolean echo) {
	private static final Log log = LogFactory.getLog(Options.class);
	public Options {
		require(modelPath != null, "Missing argument: --model <path> is required");
		require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
		require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
	}

	static final int DEFAULT_MAX_TOKENS = 512;

	static void require(boolean condition, String messageFormat, Object... args) {
		if (!condition) {
			log.error("ERROR " + messageFormat.formatted(args));
			log.info(printUsage());
			System.exit(-1);
		}
	}

	static String printUsage() {
		return 
				"Options:\r\n"+
				"  --model, -m <path>            required, path to .gguf file\r\n"+
				"  --system-prompt, -sp <string> (optional) system prompt\r\n"+
				"  --temperature, -temp <float>  temperature in [0,inf], default 0.1\r\n"+
				"  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95\r\n"+
				"  --seed <long>                 random seed, default System.nanoTime()\r\n"+
				"  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS+"\r\n"+
				"  --echo <boolean>              print ALL tokens to stderr, if true, default false\r\n"+
				"  --preload <boolean>           use the path to model file with .txt extension as context database preload\r\n"+
				"  --metadata                    write metadata file of <model file>.metadata\r\n\r\n"+
				"Examples:\r\n"+
				" --system-prompt \"Reply concisely, in French\"\r\n"+
				" --system-prompt \"Answer concisely\"\r\n";
	}

	static Options parseOptions(List<String> args) {
		String prompt = null;
		String systemPrompt = null;
		float temperature = 0.1f;
		float topp = 0.95f;
		Path modelPath = null;
		long seed = System.nanoTime();
		// Keep max context length small for low-memory devices.
		int maxTokens = DEFAULT_MAX_TOKENS;
		boolean interactive = false;
		boolean preload = false;
		boolean echo = false;
		require(2 < args.size(), "Missing argument for option %s", "all");
		for (int i = 1; i < args.size(); i++) {
			String optionName = args.get(i);
			require(optionName.startsWith("-"), "Invalid option %s", optionName);
			switch (optionName) {
			case "--interactive", "--chat", "-i" -> interactive = true;
			case "--instruct" -> interactive = false;
			case "--help", "-h" -> {
				log.info(printUsage());
				System.exit(0);
			}
			default -> {
				String nextArg;
				if (optionName.contains("=")) {
					String[] parts = optionName.split("=", 2);
					optionName = parts[0];
					nextArg = parts[1];
				} else {
					require(i + 1 < args.size(), "Missing argument for option %s", optionName);
					nextArg = args.get(i + 1);
					i += 1; // skip arg
				}
				switch (optionName) {
				case "--system-prompt", "-sp" -> systemPrompt = nextArg;
				case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
				case "--top-p" -> topp = Float.parseFloat(nextArg);
				case "--model", "-m" -> modelPath = Paths.get(nextArg);
				case "--seed", "-s" -> seed = Long.parseLong(nextArg);
				case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
				case "--echo" -> echo = Boolean.parseBoolean(nextArg);
				case "--preload" -> preload = Boolean.parseBoolean(nextArg);
				case "--metadata" -> ModelRunner.DISPLAY_METADATA = Boolean.parseBoolean(nextArg);
				default -> require(false, "Unknown option: %s", optionName);
				}
			}
			}
		}
		return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, preload, echo);
	}

}
