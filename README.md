# Royal-LLaMA: LLaMA but with jewelry

This is a fork of the LLaMA code that contains various patches that I find useful.

Current features:
- runs LLaMA-13B within 24 GiB of RAM. (adopted from [https://github.com/tloen/llama-int8])
- embedded additional samplers and repetition penalty
- implemented xformers' memory efficient attention (doesn't give any improvement on Windows on a 4090 machine, feel free to check the implementation and find the flaws)
- runs on Windows & Linux
- has an option to run IPython interactive console after it load the model. Enable it with `--ipython` cli flag

To run it on Windows, after installing requirements via pip, copy `bitsandbytes` patch to your env:
```bash
cp .\bitsandbytes_windows\*.dll <your_env_dir>\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py <your_env_dir>\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py <your_env_dir>\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```

### From [https://github.com/tloen/llama-int8]'s README:
It relies almost entirely on the `bitsandbytes` and `LLM.int8()` work of Tim Dettmers.
I've tested it on an RTX 4090, and it [reportedly works on the 3090](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454687232). It might also theoretically allow us to run LLaMA-65B on an 80GB A100, but I haven't tried this.

The code contains the following changes:

- Removes parallelism constructs
- Quantizes weights on the host machine
- Loads weights incrementally to avoid severe memory problems
- Added dependencies on `bitsandbytes`, `tqdm`.

On my Ubuntu machine with 64 GB of RAM and an RTX 4090, it takes about 25 seconds to load in the floats and quantize the model.
Users should be ready to expand their swapfiles if they don't have enough RAM.
Llamanon has also produced a [slightly uncouth user's guide](https://rentry.org/llama-tard) for using this repo, which I won't reproduce here but seems generally trustworthy.
[You may need to build `bitsandbytes` from source.](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454687232)

If you have interesting ideas for further development, I can be reached at https://twitter.com/ecjwg.

## Usage:

`python example.py --ckpt_dir [TARGET_DIR]/13B --tokenizer_path [TARGET_DIR]/tokenizer.model --max_batch_size=1`

---

# Original README

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:

```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

or the following cmd (depends on the environment)

```bash
python -m torch.distributed.run --nproc_per_node 1 example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
