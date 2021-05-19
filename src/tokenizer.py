class Tokenizer:
    def __init__(self, data):
        # data - list of words
        print('Build vocabulary')
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.token_to_idx = {
            self.pad_token: self.pad_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id,
            self.unk_token: self.unk_id,
            ' ': 4,
        }
        token_counter = 5
        for sample in data:
            tokens = list(sample)
            for token in tokens:
                if token not in self.token_to_idx:
                    self.token_to_idx[token] = token_counter
                    token_counter += 1
        self.idx_to_token = {
            value: key for (key, value) in self.token_to_idx.items()
        }
        print(f'Vocab size: {len(self.idx_to_token)}')
        self.vocab_size = len(self.idx_to_token)
    
    def encode(
        self,
        sample,
        max_length=20,
        pad_or_truncate=False,
        return_tokens=False,
        add_eos=True,
    ):
        sample_tokens = list(sample)
        if pad_or_truncate:
            if not add_eos:
                sample_tokens = sample_tokens[:max_length-1]
                sample_tokens = ['<bos>',] + sample_tokens
            else:
                sample_tokens = sample_tokens[:max_length-2]
                sample_tokens = ['<bos>',] + sample_tokens + ['<eos>',]

            sample_tokens = sample_tokens + [self.pad_token,] * (max_length - len(sample_tokens))
        if not return_tokens:
            for ind in range(len(sample_tokens)):
                if sample_tokens[ind] not in self.token_to_idx:
                    sample_tokens[ind] = self.unk_id
                else:
                    sample_tokens[ind] = self.token_to_idx[sample_tokens[ind]]
        return {
            'data': sample_tokens,
        }
    
    def decode(
        self,
        ids,
        skip_special_tokens=True,
    ):
        tokens = []
        for tok_id in ids:
            if skip_special_tokens and \
               (self.idx_to_token[tok_id] == '<pad>' or \
               self.idx_to_token[tok_id] == '<bos>' or \
               self.idx_to_token[tok_id] == '<eos>'):
               continue
            tokens.append(self.idx_to_token[tok_id])
        return tokens
