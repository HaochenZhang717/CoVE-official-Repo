import torch
from torch import nn
import random


class SpecialTokenCompressedEmbedding(nn.Module):
    def __init__(self, vocab_size, vocab_size_compressed, dim, p, a, b, channel_mean, channel_std, dtype=torch.bfloat16):
        super().__init__()
        self.vocab_size, self.vocab_size_compressed = vocab_size, vocab_size_compressed
        if self.vocab_size_compressed < self.vocab_size:
            self.embedding = nn.EmbeddingBag(vocab_size_compressed, dim, dtype=dtype)
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            self.embedding = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.p, self.a, self.b = (
            nn.Parameter(torch.LongTensor(p)[None, :], requires_grad=False),
            nn.Parameter(torch.LongTensor(a)[None, :], requires_grad=False),
            nn.Parameter(torch.LongTensor(b)[None, :], requires_grad=False),
        )

    def forward(self, x):
        if self.vocab_size_compressed < self.vocab_size:
            x_shape = x.shape
            hash_codes = ((x.flatten()[:, None] * self.a + self.b) % self.p) % self.vocab_size_compressed
            output = self.embedding(hash_codes).reshape(x_shape + (-1, ))
        else:
            output = self.embedding(x)
        return output


class SpecialTokenCompressedLMHead(nn.Module):
    def __init__(self, vocab_size, embedding_weight, p, a, b, dtype=torch.bfloat16):
        super().__init__()
        self.vocab_size, self.vocab_size_compressed, self.token_dim = vocab_size, embedding_weight.shape[0], embedding_weight.shape[1]
        self.proj = nn.Linear(self.token_dim, self.vocab_size_compressed, bias=False, dtype=dtype)
        # self.proj.weight = embedding_weight
        self.p, self.a, self.b = (
            nn.Parameter(torch.LongTensor(p)[None, :], requires_grad=False),
            nn.Parameter(torch.LongTensor(a)[None, :], requires_grad=False),
            nn.Parameter(torch.LongTensor(b)[None, :], requires_grad=False),
        )

    def forward(self, x):
        if self.vocab_size > self.vocab_size_compressed:
            hash_codes = ((torch.arange(self.vocab_size).to(x.device)[:, None] * self.a + self.b) % self.p) % self.vocab_size_compressed
            output = self.proj(x)[..., hash_codes].mean(-1)
        else:
            output = self.proj(x)
        return output


class FullTokenCompressedEmbedding(nn.Module):
    def __init__(self,
                 orig_embed: nn.Embedding,
                 item_size: int,
                 item_size_compressed: int,
                 p: int,
                 a: int,
                 b: int,
                 ):
        super().__init__()
        self.orig_embed = orig_embed
        self.orig_vocab_size = self.orig_embed.weight.shape[0]
        dim = self.orig_embed.weight.shape[1]
        dtype = self.orig_embed.weight.dtype
        channel_mean = self.orig_embed.weight.data.mean(dim=0, keepdim=True)
        channel_std = self.orig_embed.weight.data.std(dim=0, keepdim=True)
        self.item_embed = SpecialTokenCompressedEmbedding(
            item_size, item_size_compressed, dim, p, a, b, channel_mean, channel_std, dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_item = (input > (self.orig_vocab_size - 1)).to(torch.int)
        # input_remainder = torch.remainder(input, self.orig_vocab_size)
        input_remainder = input - is_item * self.orig_vocab_size
        orig_output = self.orig_embed(input_remainder * (1 - is_item)) * (1 - is_item).unsqueeze(-1)
        new_output = self.item_embed(input_remainder * is_item) * is_item.unsqueeze(-1)
        return orig_output + new_output


class FullTokenCompressedLMHead(nn.Module):
    def __init__(self, item_size, orig_embed_weight, embedding_weight, p, a, b):
        super().__init__()
        dtype = embedding_weight.dtype
        self.item_lm_head = SpecialTokenCompressedLMHead(item_size, embedding_weight, p, a, b, dtype)
        self.language_lm_head_weight = nn.Parameter(orig_embed_weight, requires_grad=False)

    def forward(self, input):
        language_out = nn.functional.linear(input, self.language_lm_head_weight)
        item_out = self.item_lm_head(input)
        return torch.cat([language_out, item_out], dim=-1)


def get_peft_embedding(model, seed, num_hashes, item_nums, compression_rate):#todo: to be tested
    p = [88579013, 24463903, 10637969, 11843057, 25772207, 90744047, 67182551, 25752593, 96196811, 99715337, 21859141,
         53504441, 47164961, 71704109, 77636203, 29129671]
    p = p[:num_hashes]
    random.seed(seed)
    a, b = [random.randint(v // 2, v) for v in p], [random.randint(0, v) for v in p]
    model.model.embed_tokens = FullTokenCompressedEmbedding(model.model.embed_tokens, item_nums,
                                                            item_nums // compression_rate, p, a, b)
    model.lm_head = FullTokenCompressedLMHead(item_nums, model.lm_head.weight,
                                              model.model.embed_tokens.item_embed.embedding.weight, p, a, b)

    # if compression_rate > 1.0:
    #     p = [88579013,24463903,10637969,11843057,25772207,90744047,67182551,25752593,96196811,99715337,21859141,53504441,47164961,71704109,77636203,29129671]
    #     p = p[:num_hashes]
    #     random.seed(seed)
    #     a, b = [random.randint(v // 2, v) for v in p], [random.randint(0, v) for v in p]
    #     model.model.embed_tokens = FullTokenCompressedEmbedding(model.model.embed_tokens, item_nums, item_nums // compression_rate, p, a, b)
    #     model.lm_head = FullTokenCompressedLMHead(item_nums, model.lm_head.weight, model.model.embed_tokens.item_embed.embedding.weight, p, a, b)
    #     # model.base_model.model.model.embed_tokens = FullTokenCompressedEmbedding(
    #     #     model.base_model.model.model.embed_tokens, item_nums, item_nums // compression_rate, p, a, b)
    #     # model.lm_head = FullTokenCompressedLMHead(
    #     #     item_nums,
    #     #     model.base_model.model.model.embed_tokens.orig_embed.weight,
    #     #     model.base_model.model.model.embed_tokens.item_embed.embedding.weight,
    #     #     p,
    #     #     a,
    #     #     b
    #     # )
    # else:
    #     model.model.embed_tokens = FullTokenEmbedding(model.model.embed_tokens, item_nums)
    #     model.lm_head = FullTokenLMHead(item_nums, model.lm_head.weight, model.model.embed_tokens.item_embed.weight)

    return model


