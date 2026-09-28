import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from kornia.color import rgb_to_lab


class LABspaceLoss(nn.Module):
    def __init__(self, losstype='deltae', eps=1e-6):
        super(LABspaceLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, output, target):
        if len(output.shape) == 2:
            output = output.unsqueeze(2).unsqueeze(3)
            target = target.unsqueeze(2).unsqueeze(3)

        output_lab = rgb_to_lab(output)
        target_lab = rgb_to_lab(target)

        output_lab[:, :1, :, :] = output_lab[:, :1, :, :]/100.0
        target_lab[:, :1, :, :] = target_lab[:, :1, :, :]/100.0
        output_lab[:, 1:, :, :] = output_lab[:, 1:, :, :]/128.0
        target_lab[:, 1:, :, :] = target_lab[:, 1:, :, :]/128.0

        L_out, a_out, b_out = output_lab[:,:1,:,:], output_lab[:,1:2,:,:], output_lab[:,2:,:,:]
        L_tgt, a_tgt, b_tgt = target_lab[:,:1,:,:], target_lab[:,1:2,:,:], target_lab[:,2:,:,:]


        if self.losstype == 'deltae':
            loss = torch.mean(torch.norm(output_lab - target_lab, dim=1))
            # print('deltaE loss', loss)

        elif self.losstype == 'lab':
            w_L, w_ab = 0.5, 1.0
            loss = w_L * torch.mean(torch.abs(L_out - L_tgt)) + w_ab * (torch.mean(torch.abs(a_out - a_tgt)) + torch.mean(torch.abs(b_out - b_tgt)))
            # print('lab loss', loss)

        elif self.losstype == 'hc':
            chroma_out = torch.sqrt(a_out **2 + b_out **2 + self.eps)
            chroma_tgt = torch.sqrt(a_tgt **2 + b_tgt **2 + self.eps)
            

            hue_out = torch.stack([a_out / chroma_out, b_out / chroma_out], dim=1)
            hue_tgt = torch.stack([a_tgt / chroma_tgt, b_tgt / chroma_tgt], dim=1)
            loss_hue = 1 - (hue_out * hue_tgt).sum(dim=1)

            loss = torch.mean(chroma_tgt * loss_hue)
        
        elif self.losstype == 'all':
            # Lightness term.
            loss_l = torch.abs(L_out - L_tgt).mean()

            # Chroma.
            chroma_out = torch.sqrt(a_out**2 + b_out**2 + self.eps)
            chroma_tgt = torch.sqrt(a_tgt**2 + b_tgt**2 + self.eps)
            loss_c = torch.abs(chroma_out - chroma_tgt).mean()

            # Hue term (skip near-neutral pixels to avoid division by ~0).
            mask = (chroma_out > 0.1) & (chroma_tgt > 0.1)
            if mask.any():
                hue_out = torch.stack([a_out/chroma_out, b_out/chroma_out], dim=1)
                hue_tgt = torch.stack([a_tgt/chroma_tgt, b_tgt/chroma_tgt], dim=1)
                loss_h = (1 - (hue_out * hue_tgt).sum(dim=1))[mask].mean()
            else:
                loss_h = torch.tensor(0.0, device=output.device)

            loss = self.w_l * loss_l + self.w_c * loss_c + self.w_h * loss_h
            
        return loss



class SparseLoss(nn.Module):
    def __init__(self, max_epochs, temperature=1.0):
        super(SparseLoss, self).__init__()
        self.min_temperature = temperature
        self.max_epochs = max_epochs
        # temperature == 1 disables the annealing.

    def forward(self, opacities_logit, epoch):
        """Opacity binary-entropy loss with a temperature that anneals over training.

        Higher temperature tolerates mid-range opacities; the temperature is
        lowered linearly from 1.0 to ``min_temperature`` as training progresses.
        """
        if epoch >= self.max_epochs:
            epoch = self.max_epochs - 1  # avoid division by zero

        progress = epoch / self.max_epochs  # 0 -> 1
        temperature = max(self.min_temperature, 1.0 - (1.0 - self.min_temperature) * progress)

        p_scaled = torch.sigmoid(opacities_logit / temperature)

        entropy = -p_scaled * torch.log(p_scaled + 1e-8) - \
                (1 - p_scaled) * torch.log(1 - p_scaled + 1e-8)
                
        return torch.mean(entropy)



## Perceptual loss that uses a pretrained VGG network
class PerceptualLoss(nn.Module):
    def __init__(self, feat_type='liu', device="cuda:0", requires_grad=False):
        super(PerceptualLoss, self).__init__()
        ## data requirement: (N,C,H,W) in RGB format, [0,1] range, and resolution >= 224x224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.feat_type = feat_type

        vgg_model = torchvision.models.vgg19(pretrained=True)

        vgg_model = vgg_model.to(device)
        if self.feat_type == 'liu':
            ## conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
            self.slice1 = nn.Sequential(*list(vgg_model.features)[:2]).eval()
            self.slice2 = nn.Sequential(*list(vgg_model.features)[2:7]).eval()
            self.slice3 = nn.Sequential(*list(vgg_model.features)[7:12]).eval()
            self.slice4 = nn.Sequential(*list(vgg_model.features)[12:21]).eval()
            self.slice5 = nn.Sequential(*list(vgg_model.features)[21:30]).eval()
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        elif self.feat_type == 'lei':
            ## conv1_2, conv2_2, conv3_2, conv4_2, conv5_2
            self.slice1 = nn.Sequential(*list(vgg_model.features)[:4]).eval()
            self.slice2 = nn.Sequential(*list(vgg_model.features)[4:9]).eval()
            self.slice3 = nn.Sequential(*list(vgg_model.features)[9:14]).eval()
            self.slice4 = nn.Sequential(*list(vgg_model.features)[14:23]).eval()
            self.slice5 = nn.Sequential(*list(vgg_model.features)[23:32]).eval()
            self.weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10.0/1.5]
        else:
            ## maxpool after conv4_4
            self.featureExactor = nn.Sequential(*list(vgg_model.features)[:28]).eval()

        self.criterion = nn.L1Loss()

        ## fixed parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.eval()
        print('[*] VGG19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def forward(self, x, y):
        ## x denotes the groundtruth; y denoets the prediction
        norm_x, norm_y = self.normalize(x), self.normalize(y)
        ## feature extract
        if self.feat_type == 'liu' or self.feat_type == 'lei':
            x_relu1, y_relu1 = self.slice1(norm_x), self.slice1(norm_y)
            x_relu2, y_relu2 = self.slice2(x_relu1), self.slice2(y_relu1)
            x_relu3, y_relu3 = self.slice3(x_relu2), self.slice3(y_relu2)
            x_relu4, y_relu4 = self.slice4(x_relu3), self.slice4(y_relu3)
            x_relu5, y_relu5 = self.slice5(x_relu4), self.slice5(y_relu4)
            x_vgg = [x_relu1, x_relu2, x_relu3, x_relu4, x_relu5]
            y_vgg = [y_relu1, y_relu2, y_relu3, y_relu4, y_relu5]
            loss = 0    
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i].detach(), y_vgg[i])
        else:
            x_vgg, y_vgg = self.featureExactor(norm_x), self.featureExactor(norm_y)
            loss = self.criterion(x_vgg.detach(), y_vgg)
        return loss