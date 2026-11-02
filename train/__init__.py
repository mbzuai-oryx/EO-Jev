"""Fine-tuning d3-flash on Earth observation.

- d3model.py      d3-flash's forward pass as a trainable model (LoRA on the text tower), imported from the shipped runtime
- data.py         training examples from the training rows: targets (one-hot, SORD, d3's own probabilities), mix, augmentation
- train.py        the training loop (KL + RPS), adapters and resumable state
- parity.py       gate R0: the trainer reproduces the runtime; gradient, overfit and merge checks
- label_replay.py d3-flash's own probabilities on the replay rows
- export.py       merge an adapter into the weights and write a package the runtime (bench/runners/d3.py) loads
"""
