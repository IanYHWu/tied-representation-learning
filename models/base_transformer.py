import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# used to give transformer positional info
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # applt sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def create_mask(seqs):
    # We mask only those vectors of the sequence in which we have all zeroes
    # (this is more scalable for some situations)
    mask = (seqs == 0).float()
    return mask[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # so the model cannot look ahead to future data
    mask = torch.triu(torch.ones((size, size)), diagonal=1).float()
    return mask  # (size, size)


def create_masks(inp, tar):
    """create all masks needed for the transformer"""

    # encoder mask
    enc_padding_mask = create_mask(inp)

    # encoder output mask
    dec_padding_mask = create_mask(inp)

    # mask for decoder inputs
    look_ahead_mask = create_look_ahead_mask(tar.size()[1])
    dec_target_mask = create_mask(tar)
    combined_mask = torch.maximum(dec_target_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_loss_mask(inp):
    """mask for the loss function"""

    mask = torch.all(inp == 0, dim=-1).float()
    return mask[:, 1:]  # (batch_size, seq_len - 1)


def scaled_dot_product_attention(q, k, v, mask):
    """
    q: query = (..., seq_len_q, depth)
    k: key = (..., seq_len_k, depth)
    v: value = (..., seq_len_v, depth_v)
    mask: float tensor with shape broadcastable to
        (..., seq_len_q, seq_len_k)

    must have seq_len_k == seq_len_v

    Returns:
        output, attention_weights
    """

    # Q @ K^T
    matmul_qk = torch.matmul(q, torch.transpose(k, -1, -2))  # (..., seq_len_q, seq_len_k)

    # (Q @ K^T) / sqrt(d_k)
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax(.)
    # (..., seq_len_q, seq_len_k)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # (Weights @ V)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v_)

    return output, attention_weights


def base_transformer_args(params):
    transformer_params = {
        'num_layers': params.layers,
        'num_heads': params.heads,
        'dff': params.dff,
        'd_model': params.d_model,
        'input_vocab_size': params.vocab_size,
        'target_vocab_size': params.vocab_size,
        'pe_input': params.max_pe,
        'pe_target': params.max_pe,
        'rate': params.dropout}

    return transformer_params


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        x : (batch_size, seq_len, d_model)
        batch_size : int

        Returns:
        x : (batch_size, num_heads, seq_len, depth)
        '''
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size()[0]

        # linear
        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # split into heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled Dot-Product Attention
        # scaled_attention (batch_size, num_heads, seq_len_q, depth)
        # attention_weights (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        # concat
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.linear(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    '''
    A pointwise feed forward network is two fully connected
    layers with a ReLU activation in between.
    '''
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        '''
        d_model = model dimension
        num_heads = number of heads
        dff = dimension of feed forward network
        rate = dropout rate
        '''
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (bath_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # enc_output shape == (batch_size, input_seq_len, d_model)

        attn1, attn_w1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_w2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm1(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_w1, attn_w2


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)

        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, self.d_model))

        self.enc_layers = nn.ModuleList([EncoderLayer(self.d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask, visualise=False):
        # x shape is batch_size x input_seq_len
        latents = []
        seq_len = x.size()[1]

        # adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # times by sqrt(d_model)

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            latents.append(x)
            x = self.enc_layers[i](x, mask)

        if visualise:
            return x, latents

        else:
            return x  # (batch_size, input_seq_len, d_model)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)

        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, self.d_model))

        self.dec_layers = nn.ModuleList([DecoderLayer(self.d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size()[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)

        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # times by sqrt(d_model)

        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_w1, attn_w2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = attn_w1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = attn_w2

        return x, attention_weights


class EncoderNoEmbed(nn.Module):
    '''Same as Encoder class above but expects embedded inputs.'''

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(EncoderNoEmbed, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, self.d_model))

        self.enc_layers = nn.ModuleList([EncoderLayer(self.d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask):
        seq_len = x.size()[1]

        # x is (batch_size, input_seq_len, d_model)

        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # times by sqrt(d_model)

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, d_model)


class DecoderNoEmbed(nn.Module):
    '''Same as Decoder class above but expects embedded inputs.'''

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(DecoderNoEmbed, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, self.d_model))

        self.dec_layers = nn.ModuleList([DecoderLayer(self.d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size()[1]
        attention_weights = {}

        # x is (batch_size, target_seq_len, d_model)

        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # times by sqrt(d_model)

        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_w1, attn_w2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = attn_w1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = attn_w2

        return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers=4, num_heads=4, dff=256,
                 d_model=64, input_vocab_size=1500, target_vocab_size=1500,
                 pe_input=1500, pe_target=1500, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_mask, look_ahead_mask, dec_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, enc_mask)

        # (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_mask)

        # (batch_size, tar_seq_len, target_vocab_size
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


if __name__ == '__main__':
    model = Transformer(1, 1, 8, 4, 7, 7, 10, 10)

    x = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 2]])
    y = torch.tensor([[1, 2, 2, 0, 3, 4], [1, 2, 2, 4, 4, 3]])

    out, attn = model(x, y, None, None, None)

    print(out)
