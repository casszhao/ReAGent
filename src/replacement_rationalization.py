import torch
import time
import math

from token_replacement.token_sampler.base import TokenSampler

@torch.no_grad()
def replacement_rationalize_lm(model, 
                   input_ids, 
                   tokenizer, 
                   replacement_token_sampler: TokenSampler,
                   verbose=True, 
                   max_steps=1024, 
                   start_step=0,
                   max_tokens_per_batch=4096):
  """Perform greedy rationalization for a language model with token replacement.

  Args:
    model: A Huggingface model. The only requirement of this model is that it
      should accept a tensor of input tokens and a tensor of corresponding 
      positions and return a vector of logits corresponding to probabilities 
      over the vocabulary. The model should be trained or fine-tuned for 
      compatibility (i.e. with word dropout) to produce sensible rationales.
    input_ids: A tensor of shape `[seq_len]` containing the input IDs of the
      sequence to rationalize.
    tokenizer: A Huggingface `Tokenizer` object that tokenizes the vocabulary.
    replacement_token_sampler: A TokenSampler (e.g. UniformTokenSampler) providing tokens for replacement
    verbose: Whether to print out the rationalization results.
    max_steps: The maximum number of steps to perform greedy rationalization.
    start_step: The first token to rationalize. This function will rationalize
      the token at step `start_step` and continue to rationalize each token 
      until the end of the sentence. The function rationalizes the entire 
      sequence by default.
    max_tokens_per_batch: The maximum number of tokens to include for the
      batching step in greedy rationalization. This should depend on the size
      of the model. If you're getting model OOM errors, try making this 
      smaller.
  
  Returns:
    all_rationales: A list of length `seq_len - 1` containing the tokens that
      form the rationales for the corresponding target tokens. The `t`-th entry
      of the list is the greedy rationale of the `t + 1`-st target token. The 
      list contains `seq_len - 1` rationales rather than `seq_len` rationales
      because we don't rationalize the first token of the sequence.
    log: A dictionary containing logging information from the rationalization 
      process.
  """
  all_rationales = []
  log = {}
  num_tokens = len(input_ids)
  start_time = time.time()

  input_text = [tokenizer.decode([token]) for token in input_ids]
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
  
  # Perform greedy rationalization for each token in the sequence, starting
  # from `start_step`.
  # prev_token = previous token of the target token
  for prev_token in range(start_step, num_tokens - 1):
    goal_word_text = input_text[prev_token + 1]
    token_log = {}
    token_log['target_position'] = prev_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []
  
    # Initialize the rationale. The rationale must always include the most
    # recent token.
    rationale = [prev_token]
    rationale_mask = torch.zeros([prev_token + 1], device=input_ids.device, dtype=input_ids.dtype)
    rationale_mask[prev_token] = 1

    # sub sequence for rationalization
    sub_sequence = input_ids[:prev_token + 1]

    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        prev_token + 1, goal_word_text))
    
    for rationale_size in range(1, min(max_steps + 1, prev_token + 2)):
      if rationale_size == 1:
        # A rationale of size 1 can only include the most recent target token.
        
        replacement_sub_sequence = replacement_token_sampler.sample(sub_sequence)
        masked_sub_sequence = rationale_mask * sub_sequence + (1 - rationale_mask) * replacement_sub_sequence
        best_logits = model(
          torch.unsqueeze(masked_sub_sequence, 0),
          position_ids=torch.unsqueeze(torch.arange(prev_token + 1), 0).to(input_ids))['logits']
        best_logits = best_logits[0, -1]
        added_token_text = input_text[prev_token]
        added_token_position = prev_token
        if verbose:
          added_token_string = ("Adding previous token to sequence: "
                                "'{}'".format(added_token_text))
      else:
        # Consider the current rationale + each target token
        candidates_new_pos = [x for x in range(prev_token + 1) 
                      if x not in rationale]
        candidates = [sorted(rationale + [x]) for x in range(prev_token + 1) 
                      if x not in rationale]
        candidate_input_ids = input_ids[[candidates]]
        candidate_position_ids = torch.tensor(candidates).to(input_ids)

        candidate_rationale_masks = rationale_mask.repeat(len(candidates_new_pos), 1)
        for i, candidate_position_ids in enumerate(candidate_position_ids):
          candidate_rationale_masks[i, candidate_position_ids] = 1
        replacement_sub_sequence = replacement_token_sampler.sample(sub_sequence)
        masked_sub_sequence = candidate_rationale_masks * sub_sequence + (1 - candidate_rationale_masks) * replacement_sub_sequence

        # Divide the candidates into batches, since all possible subsets may
        # not fit in memory if we pass them to the model at once.
        num_candidates, seq_len = candidate_input_ids.shape
        batch_size = math.floor(max_tokens_per_batch / seq_len)
        num_batches = math.ceil(num_candidates / batch_size)
        best_log_prob = -float("inf")
        for batch_ind in range(num_batches):
          batch_start_ind = batch_ind * batch_size
          batch_end_ind = (batch_ind + 1) * batch_size
          batch_input_ids = masked_sub_sequence[batch_start_ind:batch_end_ind]
          batch_position_ids = torch.arange(prev_token + 1, device=batch_input_ids.device).repeat([batch_input_ids.shape[0], 1])
          batch_logits = model(batch_input_ids, 
                               position_ids=batch_position_ids)['logits']
          # Only consider the logits for predicting the next token.
          batch_logits = batch_logits[:, -1]
          batch_log_probs = batch_logits.log_softmax(-1)[
            :, input_ids[prev_token + 1]]
          if batch_log_probs.max() > best_log_prob:
            best_log_prob = batch_log_probs.max()
            best_token = batch_log_probs.argmax() + batch_start_ind
            best_logits = batch_logits[batch_log_probs.argmax()]
        
        best_token_position = set(candidates[best_token]) - set(rationale)
        best_token_position = best_token_position.pop()
        rationale.append(best_token_position)
        rationale_mask[best_token_position] = 1

        added_token = input_text[best_token_position]
        added_token_string = "Adding token: '{}'".format(added_token)
        added_token_text = input_text[best_token_position]
        added_token_position = best_token_position
      
      best_probs = best_logits.softmax(-1)
      predicted_word_id = best_logits.argmax().item()
      predicted_word_prob = best_probs.max().item()
      predicted_word_text = tokenizer.decode([predicted_word_id])
      true_token_prob = best_probs[input_ids[prev_token + 1]].item()
      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position,
        "added_token_text": added_token_text,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "true_token_prob": true_token_prob,
      })
      if verbose:
        print("{}. This makes the top predicted word: '{}'. "
              "P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, 
                goal_word_text, true_token_prob))
      # Our combinatorial optimization is complete when the predicted token is
      # the true token.
      if torch.argmax(best_logits) == input_ids[prev_token + 1]:
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The rationale is: {}".format(
            ', '.join([input_text[x] for x in rationale])))
          print("Finished with {} tokens.".format(rationale_size))
          print("..........")
        break
    # When we've finished rationalizing, add the rationale to the complete 
    # rationale list.
    all_rationales.append(rationale)
    token_log['rationale'] = rationale
    reached_argmax = predicted_word_id == input_ids[prev_token + 1]
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalization'].append(token_log)
  
  log['all_rationales'] = all_rationales
  end_time = time.time()
  log['duration'] = end_time - start_time
  return all_rationales, log

if __name__ == "__main__":
  from transformers import AutoTokenizer, AutoModelWithLMHead
  from token_replacement import UniformTokenSampler

  # Load model from Hugging Face
  model = AutoModelWithLMHead.from_pretrained("gpt2-large")
  tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

  model.cuda()
  model.eval()

  replacement_token_sampler = UniformTokenSampler(tokenizer)

  # Generate sequence
  input_string = "I love eating breakfast out the"
  input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
  generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
  print(' generated input -->', tokenizer.decode(generated_input))
  # Rationalize sequence
  rationales, rationalization_log = replacement_rationalize_lm(model, generated_input, tokenizer, replacement_token_sampler, verbose=True)

  input_string = "She loves eating breakfast in the"
  input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
  generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
  print(' generated input -->', tokenizer.decode(generated_input))
  # Rationalize sequence
  rationales, rationalization_log = replacement_rationalize_lm(model, generated_input, tokenizer, replacement_token_sampler, verbose=True)

  input_string = "I loves cooking lunch in the"
  input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
  generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
  print(' generated input -->', tokenizer.decode(generated_input))
  # Rationalize sequence
  rationales, rationalization_log = replacement_rationalize_lm(model, generated_input, tokenizer, replacement_token_sampler, verbose=True)
