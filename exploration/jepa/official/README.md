# LeWM source reference

Unchanged source from https://github.com/lucas-maes/le-wm at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT; see LICENSE):
`module.py`, `jepa.py`, `train.py`, `utils.py`, and `config/`.
The production stochastic branch imports SIGReg, Embedder, MLP, Transformer,
and ConditionalBlock directly from `module.py`. Other files are reproduction
references, not a parallel production training path.

The encoder helper `stable_pretraining_backbone_utils.py` is unchanged from
https://github.com/galilai-group/stable-pretraining at commit
`ab836bf699a2be712dfcf980a5eb70d36391e876`, path
`stable_pretraining/backbone/utils.py` (MIT; LICENSE.stable_pretraining).
Its `vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False,
use_mask_token=False)` creates Hugging Face ViTModel with hidden_size192,
12 layers, 3 heads, intermediate_size768, and no pooling. The production model
constructs the same ViTConfig/ViTModel directly, avoiding unrelated training
framework dependencies. Transformers is pinned in requirements/jepa.txt.
No pretrained weights are used or downloaded.

The upstream LeWM train.py uses gradients through both target and context
embeddings, without EMA or detach; this branch retains that policy.
It retains the CLS-token BatchNorm MLP projector. RGB uint8 images are converted
to float/255, normalized with ImageNet mean (.485,.456,.406) and standard
deviation (.229,.224,.225), then bilinearly resized with antialiasing to 224x224.
This deterministic preprocessing follows the upstream ToImage-then-Resize order.
The upstream deterministic next-step MSE is replaced, as an explicit research
variant, by standard conditional flow matching: x_t=(1-t)epsilon+t*z_target,
t~Uniform[0,1], epsilon~N(0,I), target velocity=z_target-epsilon. Euler integration
starts at supplied noise with 10 steps by default. This is not an official Epic
implementation or MeanFlow implementation. No proprioception is used.

SIGReg is unchanged (17 knots,1024 projections,weight.09); complete batches
use the upstream T,B,D call exactly. With padded histories, invalid embeddings
are excluded and each available timestep statistic is averaged. Causal
transformer attention conditions each future token on its own action and time;
position tokens precede it in temporal order. The upstream prediction projector
is applied separately per future position (BatchNorm over B rather than B*K)
to prevent future action leakage through training statistics. Training therefore
requires at least two sequences per batch. The encoder projector retains the
upstream BatchNorm over all valid B*T images, including target images during
training; eval mode uses frozen running statistics. Invalid historical image/action
values are zeroed and never enter image normalization or SIGReg.

Source integrity is recorded in SHA256SUMS; hashes cover the retained reference
files, excluding this explanatory file and the manifest itself.
