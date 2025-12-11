package com.neocoretechs.roscar.chatformat;

import java.util.List;
import java.util.Set;

import com.neocoretechs.roscar.tokenizer.TokenizerInterface;

/**
 * Contract for all ChatFormat tokenizers
 */
public interface ChatFormatInterface {
	public TokenizerInterface getTokenizer();
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList);
	public Set<Integer> getStopTokens();
	public List<Integer> encodeHeader(ChatFormat.Message message);
	public List<Integer> encodeMessage(ChatFormat.Message message);
	/**
	 * Encode beginOfText, then follow with list of supplied messages
	 * @param appendAssistantTurn true to add a blank ASSISTANT header at the end of the list of prompts
	 * @param dialog List of messages to tokenize
	 * @return the tokenized list of appended messages
	 */
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog);
	public int getBeginOfText();
	public String stripFormatting(String input);
}
