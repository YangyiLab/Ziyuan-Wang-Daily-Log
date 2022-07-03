# Report 6.22

This report mainly records the changes of the whole gene expression profile and transcription factor expression profile after perturb for a few transcription factors. Compared with previous model, the main change is the architecture of **Decoder**. 

## Model Structure

### Encoder

Gene -> Gene -> TF (latent space)

### Decoder

**TF -> TF -> TF -> Gene**

TF -> TF -> Gene (*Previous Model*)

## Training Procedure

+ Load Pretrained Encoder
+ pre-training Decoder plus L1 regularization **epoch = 2000** (determine the topology) $\alpha = 10$ learning_rate = 1e-4
+ Mask the weights of the two networks **TF2TF** and TF2GENE in the pre-trained Decoder, that is, only retain the weights with absolute values in the top 10%. **TF2TF** consists of TF1 -> TF2 and TF2 -> TF3, we use the same topology (Mask the same path).
+ After I determine the topology, mask the rest weights and put them back to the network for training epoch = 3000, learning_rate = 1e-3 and epoch = 17000 learning_rate = 1e-4.
+ Assemble Encoder Decoder for VAE re-training to adjust weight


Decoder Code

```python
class Decoder_VAE_finetuing_bimask_decoder_2l(nn.Module):
    def __init__(self,tf2tf_mask,tf2gene_mask,n_genes=784):
        super(Decoder_VAE_finetuing_bimask_decoder_2l, self).__init__()
        self.n_genes = n_genes
        self.tf2tf_mask = tf2tf_mask
        self.tf2gene_mask = tf2gene_mask
        self.n_tfs = self.tf2tf_mask.shape[0]
        self.dropout = 0.5
        decoder = []
        decoder.append(CustomizedLinear(self.tf2tf_mask.T))
        decoder.append(nn.BatchNorm1d(self.n_tfs))
        decoder.append(nn.ReLU())
        decoder.append(nn.Dropout(self.dropout+0.2))
        self.tf2tf = nn.Sequential( *decoder)
        decoder = []
        decoder.append(CustomizedLinear(self.tf2gene_mask.T))
        decoder.append(nn.BatchNorm1d(self.n_genes))
        decoder.append(nn.ReLU())
        decoder.append(nn.Dropout(self.dropout))
        self.tf2genes = nn.Sequential( *decoder)
        
    def forward(self, z):
        z_1 = self.tf2tf(z)
        z_2 = self.tf2tf(z_1)
        x_hat = self.tf2genes(z_2)
        return x_hat,z,z_1,z_2
    
    def tf12tf2(self, z):
        z_hat = self.tf2tf(z)
        return z_hat
```

## Weight Distribution
TF2TF
```text
         0%         25%         50%         75%        100% 
-0.40189296 -0.00189805  0.00507052  0.01459043  0.82484633 
```

TF2GENE
```text
         0%         25%         50%         75%        100% 
-2.05178332 -0.00679527  0.01274097  0.03815097  2.30706573 
```