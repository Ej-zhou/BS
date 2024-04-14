```
Evaluating: Input: data/crows_pairs_anonymized.csv Model: bert ==================================================================================================== tokenizer_config.json: 100% 48.0/48.0 [00:00<00:00, 273kB/s] vocab.txt: 100% 232k/232k [00:00<00:00, 4.28MB/s] tokenizer.json: 100% 466k/466k [00:00<00:00, 6.50MB/s] config.json: 100% 570/570 [00:00<00:00, 3.92MB/s] model.safetensors: 100% 440M/440M [00:02<00:00, 198MB/s] Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight'] - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).  0% 0/1508 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)  _C._set_default_tensor_type(t) 100% 1508/1508 [09:41<00:00,  2.59it/s] 

==================================================================================================== 

Total examples: 1508 
Metric score: 60.48 
Stereotype score: 61.09 
Anti-stereotype score: 56.88 
Num. neutral: 0 0.0 ====================================================================================================
```



```
Evaluating:
Input: data/crows_pairs_trimmed.csv
Model: bert
====================================================================================================
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0% 0/1042 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)
  _C._set_default_tensor_type(t)
100% 1042/1042 [06:42<00:00,  2.59it/s]
====================================================================================================
Total examples: 1042
Metric score: 60.17
Stereotype score: 60.3
Anti-stereotype score: 59.51
Num. neutral: 0 0.0
====================================================================================================
```





```
Evaluating:
Input: data/crows_pairs_trimmed.csv
Model: mbert
====================================================================================================
tokenizer_config.json: 100% 49.0/49.0 [00:00<00:00, 185kB/s]
vocab.txt: 100% 996k/996k [00:00<00:00, 12.7MB/s]
tokenizer.json: 100% 1.96M/1.96M [00:00<00:00, 17.5MB/s]
config.json: 100% 625/625 [00:00<00:00, 1.56MB/s]
model.safetensors: 100% 714M/714M [00:04<00:00, 163MB/s]
Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0% 0/1042 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)
  _C._set_default_tensor_type(t)
100% 1042/1042 [08:06<00:00,  2.14it/s]
====================================================================================================
Total examples: 1042
Metric score: 53.65
Stereotype score: 53.7
Anti-stereotype score: 53.37
Num. neutral: 0 0.0
====================================================================================================

```



```
Evaluating:
Input: data/crows_pairs_anonymized.csv
Model: mbert
====================================================================================================
tokenizer_config.json: 100% 49.0/49.0 [00:00<00:00, 257kB/s]
vocab.txt: 100% 996k/996k [00:00<00:00, 16.3MB/s]
tokenizer.json: 100% 1.96M/1.96M [00:00<00:00, 19.3MB/s]
config.json: 100% 625/625 [00:00<00:00, 2.58MB/s]
model.safetensors: 100% 714M/714M [00:07<00:00, 98.6MB/s]
Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0% 0/1508 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)
  _C._set_default_tensor_type(t)
100% 1508/1508 [13:15<00:00,  1.90it/s]
====================================================================================================
Total examples: 1508
Metric score: 53.65
Stereotype score: 54.19
Anti-stereotype score: 50.46
Num. neutral: 0 0.0
====================================================================================================

```





```
Evaluating:
Input: data/crows_pairs_trimmed.csv
Model: xlm-roberta
====================================================================================================
Some weights of the model checkpoint at xlm-roberta-base were not used when initializing XLMRobertaForMaskedLM: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing XLMRobertaForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  0% 0/1042 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)
  _C._set_default_tensor_type(t)
100% 1042/1042 [09:02<00:00,  1.92it/s]
====================================================================================================
Total examples: 1042
Metric score: 55.66
Stereotype score: 55.18
Anti-stereotype score: 58.28
Num. neutral: 0 0.0
====================================================================================================

```

