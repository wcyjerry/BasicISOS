import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.networks.common import *
from basicseg.utils.registry import NET_REGISTRY

class Fpn_head(nn.Module):
    def __init__(self,in_planes=[64, 128, 256, 512], fpn_dim=256, scale_factor=1):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.scale_factor = 1
        self.in_planes = in_planes
        self.fpn_fuse = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        for idx in range(len(in_planes)):
            self.fpn_fuse.append(
                nn.Sequential(
                    nn.Conv2d(in_planes[idx], self.fpn_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_out.append(
                nn.Sequential(
                    nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_head = nn.Sequential(
            nn.Conv2d(len(self.in_planes) * fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fpn_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, feat_maps):
        assert len(feat_maps) == len(self.fpn_fuse)
        f = None
        fpn_outs = []
        max_shape = feat_maps[0].shape[-2:]
        for i in reversed(range(len(self.fpn_fuse))):
            shape = feat_maps[i].shape[-2:]
            fpn_fuse_out = self.fpn_fuse[i](feat_maps[i])
            if f is None:
                f = fpn_fuse_out
            else: 
                f = fpn_fuse_out + F.interpolate(f, size=shape, mode='bilinear', align_corners=False)
            fpn_outs.append(F.interpolate(self.fpn_out[i](f), size=max_shape, mode='bilinear', align_corners=False))
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_out = self.fpn_head(fpn_outs)
        return fpn_out

@NET_REGISTRY.register()
class Fpn(nn.Module):
    def __init__(self, backbone='resnet18', in_c=3, out_c=1, 
                fpn_dim=256):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = resnet18(in_c=in_c)
            inplanes = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.encoder = resnet50(in_c=in_c)
            inplanes = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.encoder = resnet101(in_c=in_c)
            inplanes = [256, 512, 1024, 2048]
        fpn_dim = fpn_dim
        self.decoder = Fpn_head(inplanes, fpn_dim)
        self.classifier = nn.Conv2d(fpn_dim, out_c, kernel_size=1)
    def forward(self, x):
        feat_maps = self.encoder(x)
        fpn_outs = self.decoder(feat_maps)
        out = self.classifier(fpn_outs)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

def main():
    import ptflops
    model = Fpn(backbone='resnet18')
    params,macs = ptflops.get_model_complexity_info(model, (3,512,512))
    print(params, macs)
    # from basicseg.utils.inference_time import test_inference_time
    # in_put = torch.rand(1,3,512,512)
    # test_inference_time(model, in_put, 3)
if __name__ == '__main__':
    main()