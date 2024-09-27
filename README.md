<h1 align="center">gLM2: A mixed-modality Genomic Language Model</h1>

<p align="center" style="font-size:0;">
  <a href="https://www.biorxiv.org/content/10.1101/2024.08.14.607850v1" style="text-decoration: none; border: none;"><img alt="bioRxiv URL" src="https://img.shields.io/badge/bioRxiv-607850v1.svg" style="border: none;"></a><a href="https://huggingface.co/tattabio/gLM2_650M" style="text-decoration: none; border: none;"><img alt="Huggingface URL" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" style="border: none;"></a>
</p>

<h4 align="center">
    <p>
        <a href="#usage">Usage</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/tattabio/gLM2_650M"><img style="float: middle;" width="120" height="120" src="./docs/images/tatta_logo.png" /></a>
</h3>

gLM2 is a mixed-modality genomic language model, trained on the [`OMG Dataset`](https://github.com/TattaBio/OMG).

The model encodes a genomic scaffold with both both amino-acid and DNA tokens. 
gLM2 is trained at two scales: [150M](https://huggingface.co/tattabio/gLM2_150M) and [650M](https://huggingface.co/tattabio/gLM2_650M) parameters.  

## Model Description
gLM2 is a transformer encoder trained with the masked language modeling objective.  
It encodes a genomic contig as a sequence of protein coding sequences (CDS) and DNA inter-genic sequences (IGS).  
CDS elements are tokenized using per-amino acid tokens, and IGS elements are tokenized using per-nucleotide tokens.  
- To encode the genomic strand, we prepended each genomic element with a special token, either `<+>` or `<->` to indicate the positive and negative strands.
- To avoid collision between amino acid and nucleotide tokens, the tokenizer expects all amino acids to be uppercase, and all nucleotides to be lowercase.

UPDATE(09/2024): We updated the model with longer context length (4096 tokens vs. 2048 tokens) and per-nucleotide IGS tokenization.

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)

# A contig with two proteins and an inter-genic sequence.
# NOTE: Nucleotides should always be lowercase, and prepended with `<+>`.
sequence = "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aatttaaggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED"

# Tokenize the sequence.
encodings = tokenizer([sequence], return_tensors='pt')
# Extract embeddings.
with torch.no_grad():
    embeddings = model(encodings.input_ids.cuda(), output_hidden_states=True).last_hidden_state
```

## Scripts

We provide a [script](https://github.com/TattaBio/gLM2/blob/main/categorical_jacobian_gLM2.ipynb) to visualize protein-protein interaction by computing the gLM2 categorical jacobian ([Zhang et al. 2024](https://www.biorxiv.org/content/10.1101/2024.01.30.577970v1)).

For example, gLM2 correctly predicts interactions between 2ONK_A (ModA) and 2ONK_C (ModC).
In comparison, ESM2 and Evo do not predict any interactions.

<p align="left">
  <img src="./docs/images/ppi_figure.png" width="669" height="364" alt="PPI Figure">
</p>

## Training Data
gLM2 is trained on the [`OMG`](https://huggingface.co/datasets/tattabio/OMG) dataset.
To improve the dataset balance and remove near-duplicate examples, the data is tokenized and pruned by applying Semantic Deduplication ([SemDedup](https://arxiv.org/abs/2303.09540)).  
We use an embedding distance threshold of 2e-3, resulting in 49% of the dataset being pruned. 



## Citing

gLM2 was introduced in "[The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling](https://www.biorxiv.org/content/10.1101/2024.08.14.607850v1)", feel free to cite:

```
@article{Cornman2024,
  title = {The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling},
  url = {https://www.biorxiv.org/content/early/2024/08/17/2024.08.14.607850},
  DOI = {10.1101/2024.08.14.607850},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Cornman, Andre and West-Roberts, Jacob and Camargo, Antonio Pedro and Roux, Simon and Beracochea, Martin and Mirdita, Milot and Ovchinnikov, Sergey and Hwang, Yunha},
  year = {2024},
}
```
