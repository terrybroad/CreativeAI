'''
Original code by Terence Broad (2024) 

This was created to make more robust and immutable configs that
can be stored as yaml files rather than python files with floating variables 
as they were originally implemented in: https://github.com/karpathy/nanoGPT/tree/master/config
'''

import typing as t
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float
    bias: bool

    @classmethod
    def from_dict(cls: t.Type["ModelConfig"], obj: dict):
        return cls(
            n_layer=obj["n_layer"],
            n_head=obj["n_head"],
            n_embd=obj["n_embd"],
            block_size=obj["block_size"],
            vocab_size=obj["vocab_size"],
            dropout=obj["dropout"],
            bias=obj["bias"],
        )

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    min_lr: float
    max_iters: int
    lr_decay_iters: int
    warmup_iters: int
    beta1: float
    beta2: float
    grad_clip: float
    weight_decay: float
    decay_lr: bool
    init_from: str

    def __post_init__(self):
        init_values = {'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-xl'}
        if self.init_from not in init_values:
            print(f'init_from must be one of either the following:{init_values}')
        
        if self.min_lr > self.learning_rate:
            print(f'min_lr {self.min_lr} should be lower than or equal to the learning_rate {self.learning_rate}')

    @classmethod
    def from_dict(cls: t.Type["TrainingConfig"], obj: dict):
        return cls(
            batch_size=obj["batch_size"],
            learning_rate=obj["learning_rate"],
            min_lr=obj["min_lr"],
            max_iters=obj["max_iters"],
            lr_decay_iters=obj["lr_decay_iters"],
            warmup_iters=obj["warmup_iters"],
            beta1=obj["beta1"],
            beta2=obj["beta2"],
            grad_clip=obj["grad_clip"],
            weight_decay=obj["weight_decay"],
            decay_lr=obj["decay_lr"],
            init_from=obj["init_from"],
        )

