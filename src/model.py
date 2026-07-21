import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A residual block for advanced U-Net architectures."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
            
    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    """
    Advanced Deep Learning Model: Res-UNet.
    Replaces standard convolutions with Residual blocks for deeper feature extraction 
    without vanishing gradients, ideal for complex spectral-temporal aggregations.
    """
    def __init__(self, in_channels, num_classes):
        super(ResUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

class TransformerBottleneck(nn.Module):
    """Transformer block for global context at the bottleneck."""
    def __init__(self, channels, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, 
            dim_feedforward=channels*4, dropout=0.1, 
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        # Apply transformer
        x_trans = self.transformer(x_flat)
        # Reshape back to (B, C, H, W)
        x_out = x_trans.permute(0, 2, 1).view(B, C, H, W)
        return x_out

class TransUNet(nn.Module):
    """
    Advanced Transformer Model: TransUNet.
    Combines local CNN feature extraction with global Transformer self-attention 
    at the bottleneck.
    """
    def __init__(self, in_channels, num_classes):
        super(TransUNet, self).__init__()
        
        # Simple Encoder
        self.enc1 = self._block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # CNN Bottleneck -> Transformer Bottleneck
        self.bottleneck_cnn = self._block(256, 512)
        self.bottleneck_trans = TransformerBottleneck(channels=512, num_heads=8, num_layers=2)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck with Transformer
        b_cnn = self.bottleneck_cnn(self.pool3(e3))
        b_trans = self.bottleneck_trans(b_cnn)
        
        d3 = self.up3(b_trans)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

class SimpleUNet(nn.Module):
    """
    A simple 2D U-Net for spatial-temporal aggregated features.
    Input shape: (B, C, H, W) where C is the number of aggregated features.
    """
    def __init__(self, in_channels, num_classes):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

def get_rf_model(config):
    """Initialize Random Forest with config params."""
    from sklearn.ensemble import RandomForestClassifier
    rf_config = config['models']['random_forest']
    return RandomForestClassifier(
        n_estimators=rf_config.get('n_estimators', 100),
        max_depth=rf_config.get('max_depth', 20),
        class_weight=rf_config.get('class_weight', 'balanced'),
        random_state=rf_config.get('random_state', 42),
        n_jobs=rf_config.get('n_jobs', -1)
    )

def get_lgbm_model(config):
    """Initialize LightGBM with config params."""
    from lightgbm import LGBMClassifier
    # We can reuse the RF config block or define reasonable LGBM defaults
    rf_config = config.get('models', {}).get('random_forest', {})
    return LGBMClassifier(
        n_estimators=rf_config.get('n_estimators', 100),
        learning_rate=0.1,
        max_depth=rf_config.get('max_depth', 20),
        class_weight='balanced',
        random_state=rf_config.get('random_state', 42),
        n_jobs=rf_config.get('n_jobs', -1),
        verbose=-1
    )

def get_unet_model(config, in_channels):
    """Initialize U-Net with config params."""
    unet_config = config['models']['unet']
    return SimpleUNet(
        in_channels=in_channels,
        num_classes=unet_config.get('num_classes', 20)
    )

def get_resunet_model(config, in_channels):
    """Initialize Advanced Res-UNet."""
    unet_config = config['models']['unet']  # reuse unet params for classes
    return ResUNet(
        in_channels=in_channels,
        num_classes=unet_config.get('num_classes', 20)
    )

def get_transunet_model(config, in_channels):
    """Initialize Transformer-based TransUNet."""
    unet_config = config['models']['unet']
    return TransUNet(
        in_channels=in_channels,
        num_classes=unet_config.get('num_classes', 20)
    )
