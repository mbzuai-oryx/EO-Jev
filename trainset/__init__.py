"""Training data for fine-tuning d3-flash on Earth observation.

Rows use the same schema as the benchmark rows (bench/common.py), so the trainer can render prompts exactly as the
runners do. Training rows also carry provenance in meta: source dataset, licence, template id, image sha256, and the
source scene as group_id.

Modules:
- hashing:    image fingerprints (sha256, perceptual hashes of the whole image and of 2×2 / 3×3 tiles)
- eval_index: fingerprints of every image in the eval suite, plus the source ids its rows record (the leak blocklist)

Everything here runs as Slurm jobs, never on the login node (slurm/convert_cpu.sbatch with MODULE=trainset.<name>).
Outputs go to $EOJEV_DATASETS/trainset/.
"""
