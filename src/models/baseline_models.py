# -*- coding: utf-8 -*-
"""
Kiến trúc cơ sở cho các mô hình Baseline (Phase 8) & Core Models (Phase 9).
Hỗ trợ các mode: video_only (K=0), history_only (Diagnostic), và full.
"""

import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    """Mã hóa thông tin tĩnh của Video (Candidate hoặc History Event)."""
    def __init__(self, num_videos, embedding_dim=64):
        super().__init__()
        # Embedding cơ bản cho video_id. 
        # Nếu ở Phase 6 ông có giữ thêm các static features khác (video_duration, author_id...)
        # thì ông khai báo thêm các lớp Embedding/Linear ở đây và concat lại nhé.
        self.video_embedding = nn.Embedding(num_embeddings=num_videos + 1, embedding_dim=embedding_dim, padding_idx=0)
        
    def forward(self, video_ids):
        # Input: (batch_size, ...) -> Output: (batch_size, ..., embedding_dim)
        return self.video_embedding(video_ids)


class MeanPoolingAggregation(nn.Module):
    """Tổng hợp lịch sử theo cơ chế Mean Pooling quy định trong Proposal."""
    def __init__(self):
        super().__init__()

    def forward(self, event_embeddings, mask):
        """
        event_embeddings: (batch_size, seq_len, embed_dim)
        mask: (batch_size, seq_len) - 1 nếu có event, 0 nếu là padding
        """
        # (batch_size, seq_len, 1)
        mask = mask.unsqueeze(-1).float()
        
        # Zero out padding embeddings
        masked_embeddings = event_embeddings * mask
        
        # Tính tổng theo sequence length
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        
        # Đếm số lượng event thực tế (tránh chia cho 0 bằng cách kẹp min = 1)
        lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        
        # Tính trung bình (Mean Pooling)
        mean_pooled = sum_embeddings / lengths
        return mean_pooled


class KuaiRandRecommender(nn.Module):
    """
    Kiến trúc linh hoạt đáp ứng mọi yêu cầu thực nghiệm từ Phase 8 đến Phase 9.
    """
    def __init__(self, num_videos, embedding_dim=64, hidden_dims=[128, 64], mode="full"):
        super().__init__()
        self.mode = mode
        assert self.mode in ["full", "video_only", "history_only"], "Mode không hợp lệ!"
        
        # 1. Encoders
        self.video_encoder = VideoEncoder(num_videos, embedding_dim)
        self.event_encoder = VideoEncoder(num_videos, embedding_dim)
        
        # 2. Aggregator
        self.aggregator = MeanPoolingAggregation()
        
        # 3. MLP Classifier
        # Kích thước đầu vào của MLP phụ thuộc vào mode
        if self.mode == "full":
            mlp_input_dim = embedding_dim * 2 # User State + Candidate Representation
        else:
            mlp_input_dim = embedding_dim     # Chỉ User State hoặc Chỉ Candidate
            
        layers = []
        input_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid()) # Dự đoán xác suất P(Long View)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate_ids, history_ids, history_mask):
        """
        candidate_ids: (batch_size,)
        history_ids: (batch_size, seq_len)
        history_mask: (batch_size, seq_len)
        """
        batch_size = candidate_ids.size(0)
        device = candidate_ids.device
        
        # --- Xử lý phần Lịch sử (User State) ---
        if self.mode == "video_only":
            # Điều kiện K=0: Ép vector history thành vector zero
            user_state = torch.zeros(batch_size, self.event_encoder.video_embedding.embedding_dim, device=device)
        else:
            # Mã hóa các sự kiện lịch sử và Mean Pooling
            event_embs = self.event_encoder(history_ids)
            user_state = self.aggregator(event_embs, history_mask)
            
        # --- Xử lý phần Candidate ---
        if self.mode == "history_only":
            # Diagnostic model: Bỏ qua thông tin candidate video
            candidate_repr = torch.empty(batch_size, 0, device=device) # Dummy tensor rỗng
        else:
            # Lấy thông tin video ứng viên
            candidate_repr = self.video_encoder(candidate_ids)
            
        # --- Kết hợp và Dự đoán (MLP) ---
        if self.mode == "full":
            combined_features = torch.cat([user_state, candidate_repr], dim=1)
        elif self.mode == "video_only":
            # Bản chất kiến trúc downstream giống nhau, nhưng user_state là zero vector
            # (Theo Proposal, để tránh đổi kiến trúc, tớ ép nó concatenate rồi tính như bình thường, 
            #  tuy nhiên để tiết kiệm parameter thì ở đây tớ truyền riêng rẽ tùy mode)
            combined_features = candidate_repr
        elif self.mode == "history_only":
            combined_features = user_state
            
        prediction = self.mlp(combined_features)
        return prediction.squeeze(-1)

# Đoạn test nhanh kiến trúc
if __name__ == "__main__":
    # Tự động chọn device tối ưu (mps cho Apple Silicon, cuda cho GPU, hoặc cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Mock data
    batch_sz, seq_len, num_vids = 4, 10, 10000
    c_ids = torch.randint(1, num_vids, (batch_sz,)).to(device)
    h_ids = torch.randint(1, num_vids, (batch_sz, seq_len)).to(device)
    h_mask = torch.ones(batch_sz, seq_len).to(device)
    
    # Test Baseline 2: Video-only learned model (K=0)
    model_k0 = KuaiRandRecommender(num_videos=num_vids, mode="video_only").to(device)
    pred_k0 = model_k0(c_ids, h_ids, h_mask)
    print(f"K=0 Output shape: {pred_k0.shape} | Mode: {model_k0.mode}")
    
    # Test Baseline 3: History-only Diagnostic
    model_hist = KuaiRandRecommender(num_videos=num_vids, mode="history_only").to(device)
    pred_hist = model_hist(c_ids, h_ids, h_mask)
    print(f"History-only Output shape: {pred_hist.shape} | Mode: {model_hist.mode}")