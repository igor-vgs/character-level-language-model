import torch


class WordsDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=20):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    @staticmethod
    def _read_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = [sample for sample in f]
        words = []
        for sentence in data[1:]:
            raw_words = sentence.split()
            raw_words = [s for s in raw_words if s not in ' !@#$%^&*()_+<>/?,.-=:;\'"']
            words.extend(raw_words)
        return words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]
        encoded_data = self.tokenizer.encode(
            word,
            max_length=self.max_length,
            pad_or_truncate=True
        )

        encoded_data['target'] = encoded_data['data'][1:] + [self.tokenizer.pad_id,]

        for key in encoded_data:
            encoded_data[key] = torch.LongTensor(encoded_data[key])
        return encoded_data
