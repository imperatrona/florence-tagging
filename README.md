Simple script to tag images for training. Saves tags in same folder with same name as files

```bash
git clone https://github.com/imperatrona/florence-tagging.git
cd florence-tagging
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash_attn # only for nvidia gpus

python main.py dataset_path
```

By default it uses https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner <DESCRIPTION> task for generating captions, you can overide by using arguments

```bash
python main.py -m "microsoft/Florence-2-base" -t "<DETAILED_CAPTION>" -p "Describe this image in great detail." dataset_path
```

Add prefix by using `--prefix` argument
```bash
pythom main.py --prefix "photo of [token]," dataset_path
```

All args:
`-m`, `--model` – florence like model path from huggingface. Default: `gokaygokay/Florence-2-SD3-Captioner`
`-t`, `--task` – prompt task. Default: `<DESCRIPTION>`
`-p`, `--prompt` – prompt for generation. Default: `Describe this image in great detail.`
`--prefix` – prefix to add in front of generated files