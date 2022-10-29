# test_transformers

```{python}
conda create -n py39 python=3.9
conda activate py39
pip install transformers
https://huggingface.co/BSC-TeMU/roberta-base-ca
https://huggingface.co/projecte-aina/roberta-base-ca-v2

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2')
model = AutoModelForMaskedLM.from_pretrained('projecte-aina/roberta-base-ca-v2')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"Em dic <mask>."
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])

https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio


```
