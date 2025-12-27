import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from ikan import ChebyKANLinear, GroupKANLinear
from timm.layers import DropPath, trunc_normal_

def get_optimal_groups(channels):
    if channels <= 8:
        return channels  
    elif channels <= 32:
        return channels // 2  
    else:
        return 32  

class GroupNormDropout(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(get_optimal_groups(num_channels), num_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return x

class DepthwiseGroupNormReLU(nn.Module):
    def __init__(self, dim=768):
        super(DepthwiseGroupNormReLU, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.gn = nn.GroupNorm(get_optimal_groups(dim), dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class GRKANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2, no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        if not no_kan:
            self.fc1 = GroupKANLinear(
                in_features=in_features,
                out_features=hidden_features,
                bias=True,
                act_mode="swish",
                drop=0.1,  
                use_conv=False,
                device=None,
                num_groups=8
            )
            self.fc2 = GroupKANLinear(
                in_features=hidden_features,
                out_features=out_features,
                bias=True,
                act_mode="swish",
                drop=0.1, 
                use_conv=False,
                device=None,
                num_groups=8
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DepthwiseGroupNormReLU(hidden_features)
        self.dwconv_2 = DepthwiseGroupNormReLU(hidden_features)
        
        self.layer_scale = nn.Parameter(torch.ones(1, 1, out_features) * 1e-5)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = x * self.layer_scale
        return x

class KolmogorovArnoldBlock(nn.Module):
    def __init__(self, dim, drop=0.2, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = GRKANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x

class KANFeatureEnhancer(nn.Module):
    def __init__(self, dim, drop=0.2, drop_path=0.1, no_kan=False):
        super(KANFeatureEnhancer, self).__init__()
        self.kan_block = KolmogorovArnoldBlock(
            dim=dim,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            no_kan=no_kan
        )
    
    def forward(self, feat):
        B, C, H, W = feat.shape
        feat_flat = feat.flatten(2).permute(0, 2, 1)
        enhanced_feat = self.kan_block(feat_flat, H, W)
        enhanced_feat = enhanced_feat.permute(0, 2, 1).reshape(B, C, H, W)
        return enhanced_feat

class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))
        
        self.freeze_bn_stats()
        
    def freeze_bn_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = True
                m.bias.requires_grad = True

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        return b1, b2, b3, b4

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        if kernel_size > 1:
            self.conv = DepthwiseSeparableConv(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
            
        self.gn = nn.GroupNorm(get_optimal_groups(out_planes), out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class SpatialPyramidAttentionBlock(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1, 3,5,7), norm_layer=None):
        super(SpatialPyramidAttentionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = lambda channels: nn.GroupNorm(get_optimal_groups(channels), channels)
        
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            nn.GroupNorm(get_optimal_groups(out_features), out_features),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.norm = nn.GroupNorm(get_optimal_groups(out_features), out_features)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_features, max(out_features//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_features//4, 8), out_features, 1),
            nn.Sigmoid()
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        
        x = self.norm(bottle)
        att = self.channel_attention(x)
        return x * att

class AdaptiveGlobalUpsample(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(AdaptiveGlobalUpsample, self).__init__()
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, 1)
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        self.norm = nn.GroupNorm(get_optimal_groups(out_channels), out_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels//4, 8), out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, high, low):
        h = self.conv_high(high)
        g = self.sigmoid(self.global_pool(h))
        l = self.conv_low(low)
        out = l * g + F.interpolate(h, size=low.size()[2:], mode='bilinear')
        
        norm_out = self.norm(out)
        att = self.channel_attention(norm_out)
        return norm_out * att

class CrossScaleAlignmentModule(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super(CrossScaleAlignmentModule, self).__init__()
        self.down_h = nn.Conv2d(in_channels_high, out_channels, 1)
        self.down_l = nn.Conv2d(in_channels_low, out_channels, 1)
        self.flow_make = DepthwiseSeparableConv(out_channels*2, 2, kernel_size=3, padding=1, bias=False)
        
    def forward(self, low_feature, high_feature):
        h_feature = self.down_h(high_feature)
        l_feature = self.down_l(low_feature)
        
        if h_feature.size()[2:] != l_feature.size()[2:]:
            h_feature = F.interpolate(h_feature, size=l_feature.size()[2:],
                                      mode='bilinear', align_corners=False)
        
        flow = self.flow_make(torch.cat([h_feature, l_feature], dim=1))
        high_feature_warped = self.feature_warping(h_feature, flow)
        
        return high_feature_warped
    
    def feature_warping(self, x, flow, size=None):
        if size is None:
            size = x.size()[2:4]
        n, c, h, w = x.size()
        
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        h_grid = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        w_grid = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=True)
        return output

class GCPB(nn.Module):
    def __init__(self, num_classes=2, feature_list=None):
        super(GCPB, self).__init__()
        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024, 2048]
        
        out_channels = 64
        
        self.context_attention = SpatialPyramidAttentionBlock(feature_list[-1], out_channels)
        
        self.upsample1 = AdaptiveGlobalUpsample(out_channels, feature_list[-2], out_channels)
        self.upsample2 = AdaptiveGlobalUpsample(out_channels, feature_list[-3], out_channels)
        self.upsample3 = AdaptiveGlobalUpsample(out_channels, feature_list[-4], out_channels)
        
        self.align1 = CrossScaleAlignmentModule(1024, out_channels, out_channels)
        self.align2 = CrossScaleAlignmentModule(512, out_channels, out_channels)
        self.align3 = CrossScaleAlignmentModule(256, out_channels, out_channels)
        
        self.dropout = nn.Dropout2d(0.3)
        
        self.norm = nn.GroupNorm(get_optimal_groups(out_channels), out_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels//4, 8), out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, c1, c2, c3, c4):
        context_out = self.context_attention(c4)
        
        aligned_context = self.align1(c3, context_out)
        fused1 = self.upsample1(context_out, c3) + aligned_context
        
        aligned_fused1 = self.align2(c2, fused1)
        fused2 = self.upsample2(fused1, c2) + aligned_fused1
        
        aligned_fused2 = self.align3(c1, fused2)
        fused3 = self.upsample3(fused2, c1) + aligned_fused2
        
        norm_out = self.norm(fused3)
        att = self.channel_attention(norm_out)
        enhanced = norm_out * att
        
        out = self.dropout(enhanced)
        return out

class AdaptiveClassificationHead(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size=3, stride=1, dilation=1, drop_out=0.3):
        super(AdaptiveClassificationHead, self).__init__()
        self.head = nn.Sequential(
            DepthwiseSeparableConv(in_feature, in_feature, kernel_size, stride,
                                  padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                                  dilation=dilation, bias=False),
            nn.GroupNorm(get_optimal_groups(in_feature), in_feature),
            nn.ReLU6(),
            nn.Dropout2d(drop_out),
            nn.Conv2d(in_feature, out_feature, kernel_size=1, bias=False))
        
    def forward(self, x):
        return self.head(x)

class EnhancedClassificationHead(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size=3, stride=1, dilation=1, drop_out=0.3):
        super(EnhancedClassificationHead, self).__init__()
        mid_channels = 128
        
        self.head = nn.Sequential(
            DepthwiseSeparableConv(in_feature, mid_channels, kernel_size, stride,
                                  padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, bias=False),
            nn.GroupNorm(get_optimal_groups(mid_channels), mid_channels),
            nn.ReLU6(),
            nn.Dropout2d(drop_out),
            nn.Conv2d(mid_channels, out_feature, kernel_size=1, bias=False))
        
    def forward(self, x):
        return self.head(x)

class SemanticAwareRefinement(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SemanticAwareRefinement, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(get_optimal_groups(out_chan), out_chan),
            nn.ReLU6())
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.gn_atten = nn.GroupNorm(get_optimal_groups(out_chan), out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        
    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.gn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class KANEnhancedSemanticDecoder(nn.Module):
    def __init__(self):
        super(KANEnhancedSemanticDecoder, self).__init__()
        out_chan = 64
        
        self.semantic_refine_16 = SemanticAwareRefinement(1024, out_chan)
        self.semantic_refine_32 = SemanticAwareRefinement(2048, out_chan)
        
        self.conv_head = nn.Sequential(
            DepthwiseSeparableConv(out_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(get_optimal_groups(out_chan), out_chan),
            nn.ReLU6())
        
        self.kan_semantic_enhancer = KANFeatureEnhancer(dim=out_chan, drop=0.2)
        
        self.norm = nn.GroupNorm(get_optimal_groups(out_chan), out_chan)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, max(out_chan//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_chan//4, 8), out_chan, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat8, feat16, feat32):
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        
        feat32_refined = self.semantic_refine_32(feat32)
        feat32_up = F.interpolate(feat32_refined, (H16, W16), mode='nearest')
        feat32_up = self.conv_head(feat32_up)

        feat16_refined = self.semantic_refine_16(feat16)
        feat16_fused = feat16_refined + feat32_up
        feat16_enhanced = self.kan_semantic_enhancer(feat16_fused)
        
        feat16_up = F.interpolate(feat16_enhanced, (H8*2, W8*2), mode='nearest')
        
        norm_out = self.norm(feat16_up)
        att = self.channel_attention(norm_out)
        return norm_out * att

class HybridGradientDetector(nn.Module):
    def __init__(self, in_channels):
        super(HybridGradientDetector, self).__init__()
        self.sobel_x_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                      groups=in_channels, bias=False)
        self.sobel_y_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                      groups=in_channels, bias=False)

        sobel_kernel_x_single = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y_single = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        sobel_kernel_x = sobel_kernel_x_single.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y_single.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

        self.sobel_x_conv.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y_conv.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

        self.edge_fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.norm = nn.GroupNorm(get_optimal_groups(in_channels), in_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels//4, 8), in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        edge_x = self.sobel_x_conv(x)
        edge_y = self.sobel_y_conv(x)
        
        fused_edge = self.edge_fusion_conv(torch.cat([edge_x, edge_y], dim=1))
        attention_map = self.sigmoid(fused_edge)
        
        enhanced_x = x * (1 + attention_map)
        
        norm_out = self.norm(enhanced_x)
        att = self.channel_attention(norm_out)
        return norm_out * att

class HybridEdgeRefinerBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, fused_edge_out_channels):
        super(HybridEdgeRefinerBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels_high, in_channels_high, kernel_size=3, groups=in_channels_high, stride=2, padding=1),
            nn.GroupNorm(get_optimal_groups(in_channels_high), in_channels_high),
            nn.ReLU(inplace=True)
        )
        
        self.flow_make = DepthwiseSeparableConv(in_channels_high * 2, 2, kernel_size=3, padding=1, bias=False)
        
        self.gradient_detector = HybridGradientDetector(in_channels_low)

        concatenated_channels = in_channels_high + in_channels_low
        self.kan_boundary_enhancer = KANFeatureEnhancer(dim=concatenated_channels, drop=0.1)
        
        self.output_conv = nn.Conv2d(concatenated_channels, fused_edge_out_channels, kernel_size=1, bias=False)

    def forward(self, x_high, x_low):
        size_high = x_high.size()[2:]
        
        seg_down = self.down(x_high)
        seg_down = F.interpolate(seg_down, size=size_high, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x_high, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x_high, flow, size_high)
        flow_edge = x_high - seg_flow_warp
        
        sobel_edge = self.gradient_detector(x_low)
        sobel_edge_upsampled = F.interpolate(sobel_edge, size=size_high, mode="bilinear", align_corners=False)
        
        concat_edges = torch.cat([flow_edge, sobel_edge_upsampled], dim=1)
        kan_enhanced_output = self.kan_boundary_enhancer(concat_edges)
        
        combined_edge = self.output_conv(kan_enhanced_output)
        
        return seg_flow_warp, combined_edge

    def flow_warp(self, input_tensor, flow, size):
        out_h, out_w = size
        n, c, h, w = input_tensor.size()

        norm = torch.tensor([[[[out_w, out_h]]]], dtype=input_tensor.dtype, device=input_tensor.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h, device=input_tensor.device).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w, device=input_tensor.device).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input_tensor, grid, align_corners=True)
        return output

class HybridEdgeGuidanceDecoder(nn.Module):
    def __init__(self, ppm_in_feat=2048, high_level_ch=32, low_level_ch=256):
        super(HybridEdgeGuidanceDecoder, self).__init__()
        self.pyramid_pooling = SpatialPyramidAttentionBlock(features=ppm_in_feat, out_features=high_level_ch, sizes=(1, 3))
        
        self.boundary_refiner = HybridEdgeRefinerBlock(
            in_channels_high=high_level_ch,
            in_channels_low=low_level_ch,
            fused_edge_out_channels=high_level_ch
        )
        
        self.edge_pred_block = nn.Sequential(
            DepthwiseSeparableConv(high_level_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_optimal_groups(16), 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False)
        )
        self.sigmoid_edge_out = nn.Sigmoid()
        
        self.seg_fusion_block = nn.Sequential(
            DepthwiseSeparableConv(high_level_ch * 2, high_level_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_optimal_groups(high_level_ch), high_level_ch),
            nn.ReLU(inplace=True)
        )
        
        self.norm = nn.GroupNorm(get_optimal_groups(high_level_ch * 2), high_level_ch * 2)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_level_ch * 2, max((high_level_ch * 2)//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max((high_level_ch * 2)//4, 8), high_level_ch * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, out_32x, out_4x):
        pooled_features = self.pyramid_pooling(out_32x)
        
        seg_body, refined_edge = self.boundary_refiner(pooled_features, out_4x)
        
        upsampled_edge = F.interpolate(refined_edge, scale_factor=8, mode='bilinear', align_corners=False)
        edge_pred_map = self.edge_pred_block(upsampled_edge)
        edge_out = self.sigmoid_edge_out(F.interpolate(edge_pred_map, scale_factor=4, mode='bilinear', align_corners=False))
        
        upsampled_seg_body = F.interpolate(seg_body, scale_factor=8, mode='bilinear', align_corners=False)
        seg_cat = torch.cat([upsampled_edge, upsampled_seg_body], dim=1)
        seg_out = self.seg_fusion_block(seg_cat)
        
        aspp_features = F.interpolate(pooled_features, scale_factor=8, mode='bilinear', align_corners=False)
        edge_feature = torch.cat([aspp_features, seg_out], dim=1)
        
        norm_out = self.norm(edge_feature)
        att = self.channel_attention(norm_out)
        return norm_out * att, edge_out

class TriBranchAdaptiveFusion(nn.Module):
    def __init__(self, in_channels):
        super(TriBranchAdaptiveFusion, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.GroupNorm(get_optimal_groups(in_channels), in_channels),
            nn.ReLU(),
        )
        
        self.msfb_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.kesb_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.hegb_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        self.norm = nn.GroupNorm(get_optimal_groups(in_channels), in_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels//4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels//4, 8), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, msfb_feat, kesb_feat, hegb_feat):
        w_msfb = self.msfb_attention(msfb_feat)
        w_kesb = self.kesb_attention(kesb_feat)
        w_hegb = self.hegb_attention(hegb_feat)
        
        weights = self.softmax(torch.cat([w_msfb, w_kesb, w_hegb], dim=1))
        
        w_msfb = weights[:, 0:1, :, :]
        w_kesb = weights[:, 1:2, :, :]
        w_hegb = weights[:, 2:3, :, :]
        
        fused_features = w_msfb*msfb_feat + w_kesb*kesb_feat + w_hegb*hegb_feat
        fused_features = self.project(fused_features)
        
        norm_out = self.norm(fused_features)
        att = self.channel_attention(norm_out)
        return norm_out * att

class GlobalContextPerceptionBranch(nn.Module):
    def __init__(self, num_class=2, feature_list=None, drop_out=0.0):
        super(GlobalContextPerceptionBranch, self).__init__()
        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024, 2048]
            
        self.feature_pyramid = GCPB(feature_list=feature_list)
        
        self.classification_head = AdaptiveClassificationHead(in_feature=64, out_feature=num_class, drop_out=drop_out)

    def forward(self, out_4x, out_8x, out_16x, out_32x):
        pyramid_out = self.feature_pyramid(out_4x, out_8x, out_16x, out_32x)
        
        semantic_out = self.classification_head(pyramid_out)
        
        return pyramid_out, semantic_out

class TKBNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TKBNet, self).__init__()
        self.num_classes = num_classes
                
        self.backbone = ResNet50()
        
        self.msfb_decoder = GlobalContextPerceptionBranch(
            num_class=2,
            feature_list=[32, 64, 128, 256, 512, 1024, 2048],
            drop_out=0.0
        )
        
        self.hegb_decoder = HybridEdgeGuidanceDecoder(
            ppm_in_feat=2048,
            high_level_ch=32,
            low_level_ch=256
        )
        
        self.kesb_decoder = KANEnhancedSemanticDecoder()
        
        self.atf_fusion = TriBranchAdaptiveFusion(in_channels=64)
        
        self.final_classifier = EnhancedClassificationHead(in_feature=64, out_feature=num_classes, drop_out=0.3)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        
        msfb_features, _ = self.msfb_decoder(x1, x2, x3, x4)
        
        hegb_features, _ = self.hegb_decoder(x4, x1)
        
        kesb_features = self.kesb_decoder(x2, x3, x4)
        
        fused_features = self.atf_fusion(msfb_features, kesb_features, hegb_features)
        
        output = self.final_classifier(fused_features)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        
        return output