from typing import List
import config
import torch


class EntityDataset:
    def __init__(
        self, texts: List[List[str]], pos: List[List[int]], tags: List[List[int]]
    ) -> None:
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item):
        text, pos, tag = self.texts[item], self.pos[item], self.tags[item]

        # tokenize for bert
        ids = []
        target_pos = []
        target_tag = []
        for i, word in enumerate(text):
            tokens = config.TOKENIZER.encode(word, add_special_tokens=False)
            ids.extend(tokens)
            # all the tokens has the same type
            target_pos.extend([pos[i]] * len(tokens))
            target_tag.extend([tag[i]] * len(tokens))
            # 2 for special token tags
            max_len = config.MAX_LEN
            padding_len = max_len - len(ids)

            ids = [101] + ids[: max_len - 2] + [102] + ([0] * padding_len)
            target_pos = [0] + target_pos[: max_len - 2] + [0] + ([0] * padding_len)
            target_tag = [0] + target_tag[: max_len - 2] + [0] + ([0] * padding_len)
            mask = ([1] * len(ids)) + ([0] * padding_len)
            token_type_ids = ([0] * len(ids)) + ([0] * padding_len)

            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "target_pos": torch.tensor(target_pos, dtype=torch.long),
                "target_tag": torch.tensor(target_tag, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }
