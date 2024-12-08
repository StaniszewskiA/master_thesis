from sklearn.metrics import precision_score, recall_score, f1_score

def evalute_tokenizer(tokenizer, reference_tokenizer, corpus) -> tuple[float, float, float]:
    tokenizer_output = [tokenizer.tokenize(word) for word in corpus]
    reference_output = [reference_tokenizer.tokenize(word) for word in corpus]

    precision = precision_score(reference_output, tokenizer_output, average='micro')
    recall = recall_score(reference_output, tokenizer_output, average='micro')
    f1 = f1_score(reference_output, tokenizer_output, average='micro')

    return precision, recall, f1

