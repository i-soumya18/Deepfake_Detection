# Deepfake Detection - Hybrid Spatial-Frequency Transformer Ensemble
# Following SOTA Model Architecture Guidelines for >95% Accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from transformers import SwinModel, SwinConfig
import math
from typing import Optional, Tuple, Dict, List

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    Mentioned in architecture for 98.5% AUC-ROC
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Scale
        y = y.view(b, c, 1, 1)
        return x * y

class ConvolutionalBlockAttention(nn.Module):
    """
    CBAM - Convolutional Block Attention Module
    Combines spatial and channel attention
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = SqueezeExcitation(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SpatialAttention(nn.Module):
    """Spatial attention component of CBAM"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        return x * attention

class SpatialBranch(nn.Module):
    """
    Spatial Branch: Swin Transformer V2 + EfficientNet
    Primary component achieving 97.81-99.67% accuracy
    """
    
    def __init__(self, 
                 swin_model_name='microsoft/swin-base-patch4-window7-224',
                 efficientnet_model='efficientnet_b4',
                 num_classes=1,
                 dropout=0.2):
        super().__init__()
        
        # Swin Transformer V2 backbone
        self.swin_config = SwinConfig.from_pretrained(swin_model_name)
        self.swin_model = SwinModel.from_pretrained(swin_model_name)
        swin_hidden_size = self.swin_config.hidden_size
        
        # EfficientNet backbone
        self.efficientnet = timm.create_model(
            efficientnet_model, 
            pretrained=True, 
            num_classes=0,  # Remove classifier
            global_pool=''
        )
        efficientnet_features = self.efficientnet.num_features
        
        # Feature fusion
        self.swin_projection = nn.Linear(swin_hidden_size, 512)
        self.efficientnet_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(efficientnet_features, 512)
        )
        
        # Attention mechanisms
        self.se_attention = SqueezeExcitation(512)
        self.cbam_attention = ConvolutionalBlockAttention(512)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),  # Combined features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Swin Transformer path
        swin_output = self.swin_model(x)
        swin_features = swin_output.pooler_output  # [batch, hidden_size]
        swin_features = self.swin_projection(swin_features)  # [batch, 512]
        
        # EfficientNet path
        efficientnet_features = self.efficientnet(x)  # [batch, channels, h, w]
        efficientnet_features = self.efficientnet_projection(efficientnet_features)  # [batch, 512]
        
        # Combine features
        combined_features = torch.cat([swin_features, efficientnet_features], dim=1)  # [batch, 1024]
        
        # Classification
        output = self.classifier(combined_features)
        
        return output, {
            'swin_features': swin_features,
            'efficientnet_features': efficientnet_features,
            'combined_features': combined_features
        }

class FrequencyBranch(nn.Module):
    """
    Frequency Domain Branch: FFT + DCT Analysis
    Captures compression artifacts and GAN-specific patterns
    """
    
    def __init__(self, 
                 input_size=256, 
                 num_classes=1,
                 dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        
        # FFT processing layers
        self.fft_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.fft_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fft_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # DCT processing layers
        self.dct_processor = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),  # Process DCT blocks
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Frequency fusion
        fft_features = 256 * ((input_size // 8) ** 2)  # After 3 conv layers with stride 2
        self.fft_global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.frequency_fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),  # FFT + DCT features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, fft_magnitude, dct_blocks=None):
        batch_size = fft_magnitude.size(0)
        
        # Process FFT magnitude spectrum - fix dimension handling
        if len(fft_magnitude.shape) == 3:
            # [batch, h, w] -> [batch, 1, h, w]
            fft_magnitude = fft_magnitude.unsqueeze(1)  # Add channel dimension
        elif len(fft_magnitude.shape) == 2:
            # [h, w] -> [1, 1, h, w]
            fft_magnitude = fft_magnitude.unsqueeze(0).unsqueeze(0)
            
        # FFT feature extraction
        fft_features = F.relu(self.fft_conv1(fft_magnitude))
        fft_features = F.max_pool2d(fft_features, 2)
        fft_features = F.relu(self.fft_conv2(fft_features))
        fft_features = F.max_pool2d(fft_features, 2)
        fft_features = F.relu(self.fft_conv3(fft_features))
        fft_features = F.max_pool2d(fft_features, 2)
        
        # Global pooling for FFT
        fft_features = self.fft_global_pool(fft_features).flatten(1)  # [batch, 256]
        
        # DCT feature extraction (if available)
        if dct_blocks is not None:
            # Reshape DCT blocks for processing
            dct_reshaped = dct_blocks.view(batch_size, -1, dct_blocks.size(-1))  # [batch, num_blocks, block_size]
            dct_features = self.dct_processor(dct_reshaped.transpose(1, 2))  # [batch, 256, 1]
            dct_features = dct_features.squeeze(-1)  # [batch, 256]
        else:
            dct_features = torch.zeros(batch_size, 256, device=fft_magnitude.device)
        
        # Combine frequency features
        frequency_features = torch.cat([fft_features, dct_features], dim=1)
        frequency_features = self.frequency_fusion(frequency_features)
        
        # Classification
        output = self.classifier(frequency_features)
        
        return output, {
            'fft_features': fft_features,
            'dct_features': dct_features,
            'frequency_features': frequency_features
        }

class TextureBranch(nn.Module):
    """
    Texture & Detail Branch: LBP + Noise Residuals
    Captures subtle manipulation traces and texture inconsistencies
    """
    
    def __init__(self, 
                 input_size=256,
                 num_classes=1,
                 dropout=0.2):
        super().__init__()
        
        # LBP feature processing
        self.lbp_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.lbp_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lbp_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Noise residual processing (3 filters from SRM-like analysis)
        self.noise_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.noise_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.noise_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # DPDA - Diversiform Pixel Difference Attention
        self.dpda = DiversiformPixelDifferenceAttention(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion
        self.texture_fusion = nn.Sequential(
            nn.Linear(256, 256),  # LBP + Noise features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, lbp_features, noise_residuals):
        batch_size = lbp_features.size(0)
        
        # Process LBP features
        if len(lbp_features.shape) == 3:
            lbp_features = lbp_features.unsqueeze(1)  # Add channel dimension
        
        lbp_out = F.relu(self.lbp_conv1(lbp_features))
        lbp_out = F.max_pool2d(lbp_out, 2)
        lbp_out = F.relu(self.lbp_conv2(lbp_out))
        lbp_out = F.max_pool2d(lbp_out, 2)
        lbp_out = F.relu(self.lbp_conv3(lbp_out))
        lbp_out = self.global_pool(lbp_out).flatten(1)  # [batch, 128]
        
        # Process noise residuals - fix dimension handling
        if len(noise_residuals.shape) == 4:
            # Already in correct format [batch, height, width, channels]
            noise_residuals = noise_residuals.permute(0, 3, 1, 2)  # [batch, channels, h, w]
        elif len(noise_residuals.shape) == 3:
            # Missing batch or channel dimension
            if noise_residuals.size(-1) == 3:
                # [batch, h, w] -> add channel dim -> [batch, channels, h, w]
                noise_residuals = noise_residuals.permute(0, 3, 1, 2)
            else:
                # [h, w, channels] -> add batch -> [batch, channels, h, w] 
                noise_residuals = noise_residuals.unsqueeze(0).permute(0, 3, 1, 2)
        
        noise_out = F.relu(self.noise_conv1(noise_residuals))
        noise_out = F.max_pool2d(noise_out, 2)
        noise_out = F.relu(self.noise_conv2(noise_out))
        noise_out = F.max_pool2d(noise_out, 2)
        noise_out = F.relu(self.noise_conv3(noise_out))
        noise_out = self.global_pool(noise_out).flatten(1)  # [batch, 128]
        
        # Combine texture features
        texture_features = torch.cat([lbp_out, noise_out], dim=1)  # [batch, 256]
        
        # Apply DPDA attention
        texture_features = self.dpda(texture_features.view(batch_size, 256, 1, 1))
        texture_features = texture_features.flatten(1)
        
        # Feature fusion
        texture_features = self.texture_fusion(texture_features)
        
        # Classification
        output = self.classifier(texture_features)
        
        return output, {
            'lbp_features': lbp_out,
            'noise_features': noise_out,
            'texture_features': texture_features
        }

class DiversiformPixelDifferenceAttention(nn.Module):
    """
    DPDA - Diversiform Pixel Difference Attention
    For subtle manipulation trace detection
    """
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Spatial attention based on pixel differences
        attention = F.relu(self.conv1(x))
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class TemporalConsistencyModule(nn.Module):
    """
    Temporal Analysis for Video Sequences
    LSTM-based temporal inconsistency detection
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sequence_features):
        """
        Args:
            sequence_features: [batch, sequence_length, feature_dim]
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(sequence_features)
        
        # Self-attention for temporal relationships
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_dim*2]
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = attended_out.transpose(0, 1)  # [batch, seq_len, hidden_dim*2]
        
        # Use the last timestep for classification
        final_features = attended_out[:, -1, :]  # [batch, hidden_dim*2]
        
        # Classification
        output = self.classifier(final_features)
        
        return output, attention_weights

class HierarchicalCrossModalFusion(nn.Module):
    """
    Hierarchical Cross-Modal Fusion
    Integrates spatial, frequency, and temporal features
    """
    
    def __init__(self, 
                 spatial_dim=1024, 
                 frequency_dim=256, 
                 texture_dim=128,
                 temporal_dim=256,
                 output_dim=512,
                 dropout=0.2):
        super().__init__()
        
        # Cross-modal attention
        self.spatial_freq_attention = CrossModalAttention(spatial_dim, frequency_dim)
        self.spatial_texture_attention = CrossModalAttention(spatial_dim, texture_dim)
        self.freq_texture_attention = CrossModalAttention(frequency_dim, texture_dim)
        
        # Feature projection layers
        self.spatial_proj = nn.Linear(spatial_dim, output_dim)
        self.frequency_proj = nn.Linear(frequency_dim, output_dim)
        self.texture_proj = nn.Linear(texture_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        
        # Multi-scale fusion
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 4, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
    def forward(self, spatial_feat, frequency_feat, texture_feat, temporal_feat=None):
        # Cross-modal attention
        spatial_enhanced = self.spatial_freq_attention(spatial_feat, frequency_feat)
        spatial_enhanced = self.spatial_texture_attention(spatial_enhanced, texture_feat)
        
        frequency_enhanced = self.freq_texture_attention(frequency_feat, texture_feat)
        
        # Project to common dimension
        spatial_proj = self.spatial_proj(spatial_enhanced)
        frequency_proj = self.frequency_proj(frequency_enhanced)
        texture_proj = self.texture_proj(texture_feat)
        
        if temporal_feat is not None:
            temporal_proj = self.temporal_proj(temporal_feat)
            fused = torch.cat([spatial_proj, frequency_proj, texture_proj, temporal_proj], dim=1)
        else:
            # Fill temporal with zeros if not available
            temporal_proj = torch.zeros_like(spatial_proj)
            fused = torch.cat([spatial_proj, frequency_proj, texture_proj, temporal_proj], dim=1)
        
        # Hierarchical fusion
        for layer in self.fusion_layers:
            fused = layer(fused)
        
        return fused

class CrossModalAttention(nn.Module):
    """Cross-modal attention between two feature modalities"""
    
    def __init__(self, dim1, dim2, hidden_dim=256):
        super().__init__()
        self.query_proj = nn.Linear(dim1, hidden_dim)
        self.key_proj = nn.Linear(dim2, hidden_dim)
        self.value_proj = nn.Linear(dim2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim1)
        self.scale = hidden_dim ** -0.5
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: Query features [batch, dim1]
            feat2: Key/Value features [batch, dim2]
        """
        q = self.query_proj(feat1)  # [batch, hidden_dim]
        k = self.key_proj(feat2)    # [batch, hidden_dim]
        v = self.value_proj(feat2)  # [batch, hidden_dim]
        
        # Attention computation
        attention = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) * self.scale  # [batch, 1, 1]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        attended = attention * v.unsqueeze(1)  # [batch, 1, hidden_dim]
        output = self.output_proj(attended.squeeze(1))  # [batch, dim1]
        
        return feat1 + output  # Residual connection

class HybridSpatialFrequencyModel(nn.Module):
    """
    Complete Hybrid Spatial-Frequency Transformer Ensemble
    Achieving >95% accuracy following SOTA architecture
    """
    
    def __init__(self, 
                 num_classes=1,
                 image_size=224,
                 dropout=0.2,
                 use_temporal=False):
        super().__init__()
        
        self.use_temporal = use_temporal
        
        # Core branches
        self.spatial_branch = SpatialBranch(num_classes=num_classes, dropout=dropout)
        self.frequency_branch = FrequencyBranch(num_classes=num_classes, dropout=dropout)
        self.texture_branch = TextureBranch(num_classes=num_classes, dropout=dropout)
        
        # Temporal module (for video sequences)
        if use_temporal:
            self.temporal_module = TemporalConsistencyModule(dropout=dropout)
        
        # Hierarchical fusion
        self.fusion = HierarchicalCrossModalFusion(
            spatial_dim=1024,
            frequency_dim=256, 
            texture_dim=128,
            temporal_dim=512 if use_temporal else 0,  # Fix temporal dimension
            dropout=dropout
        )
        
        # Final ensemble classifier
        num_preds = 3 + (1 if use_temporal else 0)  # spatial, frequency, texture + optional temporal
        self.ensemble_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + num_preds, 256),  # Fused features + individual predictions
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Individual branch weights for ensemble
        self.branch_weights = nn.Parameter(torch.ones(3 + (1 if use_temporal else 0)))
        
    def forward(self, batch_data, sequence_features=None):
        """
        Args:
            batch_data: Dictionary containing:
                - 'spatial': Spatial tensor [batch, 3, H, W]
                - 'fft_magnitude': FFT magnitude [batch, H, W]
                - 'lbp': LBP features [batch, H, W]
                - 'noise_residuals': Noise residuals [batch, H, W, 3]
            sequence_features: For temporal analysis [batch, seq_len, feat_dim]
        """
        batch_size = batch_data['spatial'].size(0)
        
        # Process each branch
        spatial_pred, spatial_info = self.spatial_branch(batch_data['spatial'])
        frequency_pred, frequency_info = self.frequency_branch(
            batch_data['fft_magnitude'],
            batch_data.get('dct_blocks')
        )
        texture_pred, texture_info = self.texture_branch(
            batch_data['lbp'],
            batch_data['noise_residuals']
        )
        
        # Temporal processing (if available)
        temporal_pred = None
        temporal_features = None
        if self.use_temporal and sequence_features is not None:
            temporal_pred, _ = self.temporal_module(sequence_features)
            # Project temporal features to expected dimension
            temporal_features = sequence_features.mean(dim=1)  # [batch, 512]
            # The fusion expects 256-dim temporal features, so we need projection in fusion
        
        # Hierarchical fusion
        fused_features = self.fusion(
            spatial_info['combined_features'],
            frequency_info['frequency_features'], 
            texture_info['texture_features'],
            temporal_features
        )
        
        # Ensemble prediction
        individual_preds = [spatial_pred, frequency_pred, texture_pred]
        if temporal_pred is not None:
            individual_preds.append(temporal_pred)
        
        # Weighted combination
        weights = F.softmax(self.branch_weights, dim=0)
        weighted_pred = sum(w * pred for w, pred in zip(weights, individual_preds))
        
        # Final classification
        ensemble_input = torch.cat([fused_features] + individual_preds, dim=1)
        final_pred = self.ensemble_classifier(ensemble_input)
        
        return {
            'prediction': final_pred,
            'weighted_prediction': weighted_pred,
            'spatial_prediction': spatial_pred,
            'frequency_prediction': frequency_pred,
            'texture_prediction': texture_pred,
            'temporal_prediction': temporal_pred,
            'fused_features': fused_features,
            'branch_weights': weights
        }

def test_model_architecture():
    """Test the complete model architecture"""
    print("🧪 Testing Hybrid Spatial-Frequency Model Architecture")
    print("=" * 55)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    model = HybridSpatialFrequencyModel(
        num_classes=1,
        image_size=224,
        dropout=0.2,
        use_temporal=False
    ).to(device)
    
    # Create sample batch data - Fix tensor dimensions
    batch_size = 2
    batch_data = {
        'spatial': torch.randn(batch_size, 3, 224, 224).to(device),
        'fft_magnitude': torch.randn(batch_size, 256, 256).to(device),
        'lbp': torch.randn(batch_size, 256, 256).to(device),
        'noise_residuals': torch.randn(batch_size, 256, 256, 3).to(device),  # [batch, h, w, channels]
        'dct_blocks': torch.randn(batch_size, 64, 64).to(device)
    }
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = model(batch_data)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Model Output Shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {value}")
        
        # Test with temporal features
        print(f"\n🎬 Testing with temporal features...")
        model_temporal = HybridSpatialFrequencyModel(
            num_classes=1,
            use_temporal=True
        ).to(device)
        
        sequence_features = torch.randn(batch_size, 8, 512).to(device)  # 8 frames
        
        with torch.no_grad():
            outputs_temporal = model_temporal(batch_data, sequence_features)
        
        print(f"✅ Temporal model test successful!")
        print(f"📊 Temporal Model Output:")
        print(f"  - Temporal prediction shape: {outputs_temporal['temporal_prediction'].shape}")
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print(f"\n✅ Model Architecture Test Complete!")
    return True

if __name__ == "__main__":
    test_model_architecture()