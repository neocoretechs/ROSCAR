package com.neocoretechs.roscar.chatformat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.neocoretechs.roscar.tokenizer.Tokenizer;
import com.neocoretechs.roscar.tokenizer.TokenizerInterface;

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
public class ChatFormat implements ChatFormatInterface {

	final Tokenizer tokenizer;
	final int beginOfText;
	final int endHeader;
	final int startHeader;
	final int endOfTurn;
	final int endOfText;
	final int endOfMessage;
	final Set<Integer> stopTokens;

	public ChatFormat(TokenizerInterface tokenizer) {
		this.tokenizer = (Tokenizer)tokenizer;
		Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
		this.beginOfText = specialTokens.get("<|begin_of_text|>");
		this.startHeader = specialTokens.get("<|start_header_id|>");
		this.endHeader = specialTokens.get("<|end_header_id|>");
		this.endOfTurn = specialTokens.get("<|eot_id|>");
		this.endOfText = specialTokens.get("<|end_of_text|>");
		this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
		this.stopTokens = Set.of(endOfText, endOfTurn);
	}
	@Override
	public TokenizerInterface getTokenizer() {
		return tokenizer;
	}
	@Override
	public Set<Integer> getStopTokens() {
		return stopTokens;
	}
	@Override
	public int getBeginOfText() {
		return beginOfText;
	}
	@Override
	public List<Integer> encodeHeader(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(startHeader);
		tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
		tokens.add(endHeader);
		tokens.addAll(this.tokenizer.encodeAsList("\n"));
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
		tokens.add(endOfTurn);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(tokenList);
		tokens.add(endOfTurn);
		return tokens;
	}
	/**
	 * Encode beginOfText, then follow with list of supplied messages
	 * @param appendAssistantTurn true to add a blank ASSISTANT header at the end of the list of prompts
	 * @param dialog List of messages to tokenize
	 * @return the tokenized list of appended messages
	 */
	@Override
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(beginOfText);
		for (ChatFormat.Message message : dialog) {
			tokens.addAll(this.encodeMessage(message));
		}
		if (appendAssistantTurn) {
			// Add the start of an assistant message for the model to complete.
			tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
		}
		return tokens;
	}
	@Override
	public String stripFormatting(String input) {
		return input.replaceAll("<\\|.*?\\|>", "")
				.replaceAll("\\*+", "")
				.replaceAll("(?m)^USER:|AI:", "")
				.trim();
	}

	public record Message(ChatFormat.Role role, String content) {
	}

	public enum Role {
		SYSTEM("SYSTEM"),
		USER("USER"),
		ASSISTANT("ASSISTANT");
		private final String role;
		Role(String role) {
			this.role = role;
		}
		public String getRole() {
			return role;
		}
		@Override
		public String toString() {
			return role;
		}
	}
}
