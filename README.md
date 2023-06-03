---
license: apache-2.0
tags:
- generated_from_keras_callback
model-index:
- name: 8CSI/Travel_Sentimental_Analysis
  results: []
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# 8CSI/Travel_Sentimental_Analysis

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Train Loss: 0.0971
- Train Accuracy: 0.9690
- Validation Loss: 0.1290
- Validation Accuracy: 0.9569
- Epoch: 1

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'Adam', 'weight_decay': None, 'clipnorm': 1.0, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': True, 'is_legacy_optimizer': False, 'learning_rate': 3e-05, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'amsgrad': False}
- training_precision: float32

### Training results

| Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Epoch |
|:----------:|:--------------:|:---------------:|:-------------------:|:-----:|
| 0.1391     | 0.9523         | 0.1285          | 0.9551              | 0     |
| 0.0971     | 0.9690         | 0.1290          | 0.9569              | 1     |


### Framework versions

- Transformers 4.29.2
- TensorFlow 2.12.0
- Datasets 2.12.0
- Tokenizers 0.13.3
