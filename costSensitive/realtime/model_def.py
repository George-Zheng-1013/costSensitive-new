from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BYTE_PAD_TOKEN = 256
BYTE_VOCAB_SIZE = 257


class PacketEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=BYTE_VOCAB_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=BYTE_PAD_TOKEN,
        )
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, packet_bytes: torch.Tensor) -> torch.Tensor:
        # packet_bytes: [N, packet_len]
        x = self.embedding(packet_bytes)  # [N, packet_len, emb]
        x = x.transpose(1, 2)  # [N, emb, packet_len]
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool(x).squeeze(-1)  # [N, 128]
        return self.proj(x)  # [N, out_dim]


class SessionEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_size: int = 128):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self, packet_features: torch.Tensor, packet_mask: torch.Tensor
    ) -> torch.Tensor:
        # packet_features: [B, num_packets, feat_dim]
        # packet_mask: [B, num_packets] True means real packet.
        lengths = packet_mask.long().sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            packet_features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=packet_features.size(1),
        )

        mask = packet_mask.unsqueeze(-1).to(out.dtype)
        summed = (out * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / denom  # [B, 2*hidden_size]


class ByteSessionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        packet_feature_dim: int = 128,
        rnn_hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.packet_encoder = PacketEncoder(out_dim=packet_feature_dim)
        self.session_encoder = SessionEncoder(
            input_dim=packet_feature_dim,
            hidden_size=rnn_hidden_size,
        )
        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, bytes: torch.Tensor, packet_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # bytes: [B, num_packets, packet_len], dtype long
        # packet_mask: [B, num_packets], dtype bool
        bsz, num_packets, packet_len = bytes.shape
        flat_bytes = bytes.reshape(bsz * num_packets, packet_len)
        packet_feat = self.packet_encoder(flat_bytes)
        packet_feat = packet_feat.reshape(bsz, num_packets, -1)
        session_feat = self.session_encoder(packet_feat, packet_mask)
        embedding = self.embed_head(session_feat)
        logits = self.classifier(embedding)
        return logits, embedding

    @torch.no_grad()
    def predict(self, bytes: torch.Tensor, packet_mask: torch.Tensor):
        logits, _ = self.forward(bytes, packet_mask)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_label = probs.max(dim=-1)
        return pred_label, confidence

    @torch.no_grad()
    def extract_embedding(self, bytes: torch.Tensor, packet_mask: torch.Tensor):
        _, embedding = self.forward(bytes, packet_mask)
        return embedding
