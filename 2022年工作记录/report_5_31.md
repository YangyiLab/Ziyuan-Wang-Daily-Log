# Report 5.31

This report mainly records the changes of the whole gene expression profile and transcription factor expression profile after perturb for a few transcription factors. Compared with the previous model, when a transcription factor is changed, more transcription factors are changed and more genes are also changed, but this change is still difficult to see after UMAP dimension reduction. 

It should be pointed out that only Gata1 and Spi1, two transcription factors, carried out perturbation, but only Spi1 carried out perturbation in the perturbation from CMP to GMP, **because the expression amount of Gata1 did not change significantly in the two kinds of cells (CMP VS GMP).**

## Model Structure

### Encoder

Gene -> Gene -> TF (latent space)

### Decoder

TF -> TF -> Gene

## Training Procedure

+ Pre-train Encoder
+ pre-training Decoder plus L2 regularization epoch = 2000 (determine the topology)
+ Mask the weights of the two networks TF2TF and TF2GENE in the pre-trained Decoder, that is, only retain the weights with absolute values in the top 10%
+ After each weight *5, mask the rest weights and put them back to the network for pre-training epoch = 10000
+ Assemble Encoder Decoder for VAE re-training to adjust weight


## Hyper Parametre

+ L2 regularization weight 1e-4
+ learning rate 1e-4
+ Early stop patience 20

## Apeendix

+ DE Map
+ TF2TF (VAE.Decoder) weight distribution map
+ TF2GENE (VAE.Decoder) weight distribution map