```
Evaluating: Input: data/crows_pairs_anonymized.csv Model: bert ==================================================================================================== tokenizer_config.json: 100% 48.0/48.0 [00:00<00:00, 273kB/s] vocab.txt: 100% 232k/232k [00:00<00:00, 4.28MB/s] tokenizer.json: 100% 466k/466k [00:00<00:00, 6.50MB/s] config.json: 100% 570/570 [00:00<00:00, 3.92MB/s] model.safetensors: 100% 440M/440M [00:02<00:00, 198MB/s] Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight'] - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).  0% 0/1508 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/__init__.py:696: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:451.)  _C._set_default_tensor_type(t) 100% 1508/1508 [09:41<00:00,  2.59it/s] 

==================================================================================================== 

Total examples: 1508 

Metric score: 60.48 

Stereotype score: 61.09 

Anti-stereotype score: 56.88 

Num. neutral: 0 0.0 ====================================================================================================
```

