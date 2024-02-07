import gc
import json
import time

import torch
from flask import Flask
from flask import jsonify
from flask import request
from transformers import AutoTokenizer, AutoModelForCausalLM



def to_tokens_and_logprobs(model, tokenizer, input_texts,device,special_ids):
    """

    :param model:
    :param tokenizer:
    :param input_texts:
    :return: [[('One', -5.882715702056885),
  (' plus', -9.785109519958496),
  (' one', -0.7229145169258118),
  (' is', -2.494063377380371),
  (' two', -6.137458324432373)],
]
    """
    # print(input_texts)
    # print(input_ids,other)
    input_ids = tokenizer(input_texts,
                          max_length=1024,
                          return_tensors="pt"
                          ).input_ids.to(device)

    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1::]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    tokens=[]
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids+special_ids:
                text_sequence.append((tokenizer.decode(token.item()), p.item()))
                # tokens.append(token.cpu())
                tokens.append(tokenizer.decode(token.item()))
        batch.append(text_sequence)

    batch = batch[0]
    logprobs = [x[1] for x in batch]
    top_logprobs_dicts = [[{x[0].strip(): x[1]} for x in batch]]
    del input_ids, outputs, probs
    torch.cuda.empty_cache()
    print("to_tokens_and_logprobs",logprobs,top_logprobs_dicts)
    return [tokens],logprobs, top_logprobs_dicts


def convert_tokens(text, tokenizer):
    """
    text to tokens
    :param text:
    :param tokenizer:
    :return:
    """
    input_ids = tokenizer(text)['input_ids']
    tokens = []
    for id in input_ids:
        tokens.append(tokenizer.decode(id))
    return tokens
