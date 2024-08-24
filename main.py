import argparse
import glob
import os
import re
from unittest.mock import patch

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

parser = argparse.ArgumentParser(
    prog="FlorenceTagging",
    description="Use florence based models for auto tagging training datasets",
)

parser.add_argument("path")
parser.add_argument("-m", "--model", default="gokaygokay/Florence-2-SD3-Captioner")
parser.add_argument("-t", "--task", default="<DESCRIPTION>")
parser.add_argument("-p", "--prompt", default="Describe this image in great detail.")
parser.add_argument("--prefix", default="")

args = parser.parse_args()

paths = []
types = ("*.jpg", "*.png")

if os.path.isfile(args.path):
    paths.append(args.path)
if os.path.isdir(args.path):
    for files in types:
        paths.extend(glob.glob(os.path.join(args.path, files)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def load_model(model_path):
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        return (model, processor)


model, processor = load_model(args.model)

for p in paths:
    try:
        with Image.open(p) as img:
            inputs = processor(
                text=args.task + args.prompt, images=img, return_tensors="pt"
            ).to(device, torch_dtype)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed_answer = processor.post_process_generation(
                generated_text,
                task=args.task,
                image_size=(img.width, img.height),
            )

            result_path = re.sub(r".\w+$", ".txt", p)
            with open(result_path, "w") as result_file:
                result_file.write(args.prefix + parsed_answer[args.task])

            print(args.prefix + parsed_answer[args.task])
    except UnidentifiedImageError:
        print(f"Skiped cause error happend while trying to open {p}")
