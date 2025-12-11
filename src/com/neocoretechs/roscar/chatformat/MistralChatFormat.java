package com.neocoretechs.roscar.chatformat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.neocoretechs.roscar.tokenizer.TokenizerInterface;

/**
 * Utility tailored for Mistral v0.3 instruct prompt format.
 */
public final class MistralChatFormat implements ChatFormatInterface {

	protected final TokenizerInterface tokenizer;
	protected final int unknownToken;
	protected final int beginOfText;
	protected final int endOfText;
	protected final int beginOfInstruction;
	protected final int endOfInstruction;
	protected final int toolCalls;
	protected final int beginOfAvailableTools;
	protected final int endOfAvailableTools;
	protected final int beginOfToolResults;
	protected final int endOfToolResults;
	protected final int prefix;
	protected final int middle;
	protected final int suffix;

	public MistralChatFormat(TokenizerInterface tokenizer) {
		this.tokenizer = tokenizer;
		Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
		this.unknownToken = specialTokens.get("<unk>");
		this.beginOfText = specialTokens.get("<s>");
		this.endOfText = specialTokens.get("</s>");
		this.beginOfInstruction = specialTokens.get("[INST]");
		this.endOfInstruction = specialTokens.get("[/INST]");
		this.toolCalls = specialTokens.get("[TOOL_CALLS]");
		this.beginOfAvailableTools = specialTokens.get("[AVAILABLE_TOOLS]");
		this.endOfAvailableTools = specialTokens.get("[/AVAILABLE_TOOLS]");
		this.beginOfToolResults = specialTokens.get("[TOOL_RESULTS]");
		this.endOfToolResults = specialTokens.get("[/TOOL_RESULTS]");
		// Only Codestral supports FIM tokens.
		this.prefix = specialTokens.getOrDefault("[PREFIX]", unknownToken);
		this.suffix = specialTokens.getOrDefault("[SUFFIX]", unknownToken);
		this.middle = specialTokens.getOrDefault("[MIDDLE]", unknownToken);
	}
	@Override
	public TokenizerInterface getTokenizer() {
		return tokenizer;
	}
	@Override
	public Set<Integer> getStopTokens() {
		return Set.of(endOfText);
	}
	@Override
	public int getBeginOfText() {
		return beginOfText;
	}

	public List<Integer> encodeMessage(String userMessage, boolean addHeader, boolean addFooter) {
		List<Integer> tokens = new ArrayList<>();
		if (addHeader) {
			tokens.add(this.beginOfInstruction);
		}
		if (userMessage != null) {
			tokens.addAll(this.tokenizer.encodeAsList(userMessage.strip()));
		}
		if (addFooter) {
			tokens.add(endOfInstruction);
		}
		return tokens;
	}

	public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.suffix);
		tokens.addAll(tokenizer.encode(suffix));
		tokens.add(this.prefix);
		tokens.addAll(tokenizer.encode(prefix));
		return tokens;
	}
	@Override
	public List<Integer> encodeHeader(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(tokenList);
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(beginOfText);
		for (ChatFormat.Message message : dialog) {
			tokens.addAll(this.encodeMessage(message));
		}
		//if (appendAssistantTurn) {
		//    // Add the start of an assistant message for the model to complete.
		//    tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
		//}
		tokens.add(endOfText);
		return tokens;
	}
	@Override
	public String stripFormatting(String input) {
		return input
				.replaceAll("\\[/?[A-Z_]+\\]", "")
				.replaceAll("\\[(PREFIX|SUFFIX|MIDDLE)\\]", "")
				.replaceAll("<\\|.*?\\|>", "")
				.replaceAll("\\*+", "")
				.replaceAll("(?m)^USER:|AI:|ASSISTANT:", "")
				.trim();
	}
}

