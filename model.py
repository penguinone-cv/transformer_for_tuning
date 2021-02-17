import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    '''idで示されている単語をベクトルに変換します'''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        # freeze=Trueによりバックプロパゲーションで更新されず変化しなくなります

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

class MakeTokens(nn.Module):
    def __init__(self, tokens):
        super(MakeTokens, self).__init__()
        self.tokens = tokens
        self.d_model = 28*28 // tokens

    def forward(self, x):
        batch_size = len(x)
        x = x.view(batch_size, self.tokens, self.d_model)   #Linear Tokenize
        return x


class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトル情報を付加する'''

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数
        self.tokens = max_seq_len

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))

                # 誤植修正_200510 #79
                # pe[pos, i + 1] = math.cos(pos /
                #                          (10000 ** ((2 * (i + 1))/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        #ret = math.sqrt(self.d_model)*x + self.pe
        ret = x + self.pe.view(-1, self.tokens, self.d_model)
        return ret


class Attention(nn.Module):
    '''Attention1つ分'''

    def __init__(self, d_model=300):
        super().__init__()

        # SAGANでは1dConvを使用したが、今回は全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        #mask = mask.unsqueeze(1)
        #weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output#, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニット'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.attention_models = nn.ModuleList()
        self.heads = heads
        for i in range(heads):
            self.attention_models.append(Attention(d_model))

        self.linear = nn.Linear(d_model*heads, d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model, d_ff)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)

        attentions = []
        for i, model in enumerate(self.attention_models):
            out = model(x_normlized, x_normlized, x_normlized, mask)
            attentions.append(out)
            #print(out.to('cpu').detach().numpy().copy().shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output = torch.cat(attentions, dim=2).to(device)
        #print(output.to('cpu').detach().numpy().copy().shape)
        output = self.linear(output)


        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output#, normlized_weights


class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量（300次元）を取り出す
        out = self.linear(x0)

        return out

class TransformerClassification(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, heads, layers_num, dropout_rate=0.1, d_model=1, d_ff=1024, max_seq_len=784, output_dim=10):
        super().__init__()

        # モデル構築
        #self.embedder = Embedder(text_embedding_vectors)
        self.maketokens = MakeTokens(max_seq_len)
        #self.position_add = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.layers_num = layers_num
        self.encoders = nn.ModuleList()
        for i in range(layers_num):
            self.encoders.append(TransformerBlock(d_model, heads, d_ff, dropout_rate))
        #self.net3_1 = TransformerBlock(d_model=d_model)
        #self.net3_2 = TransformerBlock(d_model=d_model)
        self.classifier = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        #x = self.embedder(x)  # 単語をベクトルに
        x = self.maketokens(x)
        #x = self.position_add(x)  # Positon情報を足し算
        for i in range(self.layers_num):
            x = self.encoders[i](x, mask)
        #x3_1, normlized_weights_1 = self.net3_1(x2, mask)  # Self-Attentionで特徴量を変換
        #x3_2, normlized_weights_2 = self.net3_2(x3_1, mask)  # Self-Attentionで特徴量を変換
        x = self.classifier(x)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        return x#, normlized_weights_1, normlized_weights_2
