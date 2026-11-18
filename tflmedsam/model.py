"""Frozen image encoder, residual adapters, text fusion, and mask refinement."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self,width,rank):
        super().__init__();self.down=nn.Linear(width,rank,bias=False);self.up=nn.Linear(rank,width,bias=False)
        self.scale=nn.Parameter(torch.ones(()));nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5));nn.init.zeros_(self.up.weight)
    def forward(self,x): return self.scale*self.up(F.gelu(self.down(x)))

class AdaptedFFN(nn.Module):
    def __init__(self,ffn,width,rank): super().__init__();self.ffn=ffn;self.adapter=Adapter(width,rank)
    def forward(self,x): return self.ffn(x)+self.adapter(x)

class Interaction(nn.Module):
    def __init__(self,c,h):
        super().__init__();self.ti=nn.MultiheadAttention(c,h,batch_first=True);self.it=nn.MultiheadAttention(c,h,batch_first=True)
        self.tn=nn.LayerNorm(c);self.xn=nn.LayerNorm(c)
    def forward(self,x,t,padding):
        t=self.tn(t+self.ti(t,x,x,need_weights=False)[0])
        x=self.xn(x+self.it(x,t,t,key_padding_mask=padding,need_weights=False)[0]);return x,t

class TF_LMedSAM(nn.Module):
    def __init__(self,encoder,text_width,channels=256,rank=64,heads=8,layers=2,use_adapter=True,use_text=True):
        super().__init__()
        if len(encoder.blocks)!=12: raise ValueError('The encoder must have 12 layers')
        for p in encoder.parameters(): p.requires_grad_(False)
        self.patch_embed=encoder.patch_embed;self.pos_embed=encoder.pos_embed;self.blocks=encoder.blocks
        width=self.pos_embed.shape[-1];self.use_text=use_text
        if use_adapter:
            for b in self.blocks: b.mlp=AdaptedFFN(b.mlp,width,rank)
        self.projections=nn.ModuleList([nn.Conv2d(width,channels,1) for _ in range(4)])
        self.low_projection=nn.Conv2d(width,channels,1)
        self.text_projection=nn.Linear(text_width,channels)
        self.fusion=nn.MultiheadAttention(channels,heads,batch_first=True)
        self.interactions=nn.ModuleList([Interaction(channels,heads) for _ in range(layers)])
        self.query_projection=nn.Linear(channels,channels)
        # Equation (5): upsample the coarse mask, concatenate low features,
        # and apply a 1x1 convolution. A final resize restores input dimensions.
        self.refinement=nn.Conv2d(channels+1,1,1)
        self.class_queries=nn.Parameter(torch.randn(4,channels)*.02,requires_grad=not use_text)
        self.register_buffer('mean',torch.tensor([123.675,116.280,103.530])[None,:,None,None])
        self.register_buffer('std',torch.tensor([58.395,57.120,57.375])[None,:,None,None])
    def encode(self,images):
        x=self.patch_embed((images*255-self.mean)/self.std)
        if x.shape[1:]!=self.pos_embed.shape[1:]: raise ValueError('Input dimensions do not match encoder positional embeddings')
        x=x+self.pos_embed;taps=[];low=None
        for i,b in enumerate(self.blocks,1):
            x=b(x)
            if i==1: low=x.permute(0,3,1,2)
            if i in (3,6,9,12): taps.append(x.permute(0,3,1,2))
        size=taps[0].shape[-2:];maps=[]
        for i,(tap,projection) in enumerate(zip(taps,self.projections)):
            f=projection(tap);factor=2**i
            f=F.adaptive_avg_pool2d(f,(max(1,size[0]//factor),max(1,size[1]//factor)))
            maps.append(F.interpolate(f,size=size,mode='bilinear',align_corners=False))
        return sum(maps)/4,self.low_projection(low)
    def forward(self,images,tokens,global_features,token_valid):
        visual,low=self.encode(images);b,c,h,w=visual.shape;outputs=[]
        for k in range(4):
            x=visual.flatten(2).transpose(1,2)
            if self.use_text:
                # Global EOS is a decoder query; token states alone supply
                # fusion keys/values, preserving the L-token attention shape.
                words=self.text_projection(tokens[k])[None].expand(b,-1,-1)
                valid=token_valid[k][None].expand(b,-1);padding=~valid
                x=x+self.fusion(x,words,words,key_padding_mask=padding,need_weights=False)[0]
                query=self.text_projection(global_features[k])[None,None].expand(b,1,-1)
                decoder_padding=torch.zeros((b,1),dtype=torch.bool,device=x.device)
            else:
                query=self.class_queries[k][None,None].expand(b,1,-1)
                decoder_padding=torch.zeros((b,1),dtype=torch.bool,device=x.device)
            for layer in self.interactions: x,query=layer(x,query,decoder_padding)
            features=x.transpose(1,2).reshape(b,c,h,w)
            coarse_features=F.adaptive_avg_pool2d(features,(max(1,h//2),max(1,w//2)))
            q=self.query_projection(query[:,0])
            coarse=torch.einsum('bchw,bc->bhw',coarse_features,q)[:,None]/math.sqrt(c)
            coarse_up=F.interpolate(coarse,size=low.shape[-2:],mode='bilinear',align_corners=False)
            outputs.append(self.refinement(torch.cat([coarse_up,low],1)))
        return F.interpolate(torch.cat(outputs,1),size=images.shape[-2:],mode='bilinear',align_corners=False)

def loss_function(logits,target,dice_weight=1.,bce_weight=1.,alpha=1.,beta=.1):
    p=torch.sigmoid(logits);bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    q=p.clamp(1e-7,1-1e-7);entropy=-q*torch.log(q)-(1-q)*torch.log1p(-q)
    dice=1-(2*(p*target).sum((0,2,3))+1e-6)/((p+target).sum((0,2,3))+1e-6)
    return dice_weight*dice.mean()+bce_weight*bce.mean()+beta*((1+alpha*entropy)*bce).mean()

class CheckBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
    def forward(self,x):
        shape=x.shape;y=x.reshape(shape[0],-1,shape[-1]);n=self.n1(y)
        y=y+self.attn(n,n,n,need_weights=False)[0];return (y+self.mlp(self.n2(y))).reshape(shape)

class CheckPatch(nn.Module):
    def __init__(self,width):super().__init__();self.conv=nn.Conv2d(3,width,8,stride=8)
    def forward(self,x):return self.conv(x).permute(0,2,3,1)

class CheckEncoder(nn.Module):
    def __init__(self,size=64,width=32):
        super().__init__();self.patch_embed=CheckPatch(width)
        self.pos_embed=nn.Parameter(torch.zeros(1,size//8,size//8,width))
        self.blocks=nn.ModuleList([CheckBlock(width) for _ in range(12)])

class FrozenLinearSegmenter(nn.Module):
    """Frozen encoder plus a linear 1x1 decoder, optionally with adapters."""
    def __init__(self,encoder,rank=64,use_adapter=False):
        super().__init__()
        for p in encoder.parameters():p.requires_grad_(False)
        self.patch_embed=encoder.patch_embed;self.pos_embed=encoder.pos_embed;self.blocks=encoder.blocks
        width=self.pos_embed.shape[-1]
        if use_adapter:
            for b in self.blocks:b.mlp=AdaptedFFN(b.mlp,width,rank)
        self.head=nn.Conv2d(width,4,1)
        self.register_buffer('mean',torch.tensor([123.675,116.280,103.530])[None,:,None,None])
        self.register_buffer('std',torch.tensor([58.395,57.120,57.375])[None,:,None,None])
    def forward(self,images,*unused_text):
        x=self.patch_embed((images*255-self.mean)/self.std)+self.pos_embed
        for b in self.blocks:x=b(x)
        return F.interpolate(self.head(x.permute(0,3,1,2)),size=images.shape[-2:],mode='bilinear',align_corners=False)
