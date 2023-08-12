from diffusers import StableDiffusionXLPipeline
import torch
import os
import os.path as osp

# train 
'''
accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="oneshot-dataset/golden-flower" \
  --learnable_property="style" \
  --content_prompt="a flower" \
  --placeholder_token="<melting-golden-3d-rendering>" --initializer_token="golden" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sdxl-ti-melting-golden" \
--save_as_full_pipeline \
--seed=0

'''

config_dict = {
    "objects": ["baby penguin", "moose", "towel", "letter 'G'", "robot", "crown"],
    "style": "<melting-golden-3d-rendering>",
    "model_path": "ti-sdxl-melting-golden",
    "save_image_dir": "imgs-ti-sdxl",
    "gpu_id": 3,
    "num_inference_steps": 30,
    "cfg": 7.5
}

# load model
pipe = StableDiffusionXLPipeline.from_pretrained(config_dict["model_path"], torch_dtype=torch.float16)
pipe.to("cuda:{}".format(config_dict["gpu_id"]))

# prompt as "A {object} in {style}"
objects = config_dict["objects"]
style = config_dict["style"]
prompts = [
    f"A {object} in the style of {style}" for object in objects
]

# save images dir
root_dir = "{}_{}".format(config_dict["save_image_dir"], style.replace(" ", "_"))
if not osp.isdir(root_dir):
    os.mkdir(root_dir)
paths = ["{}/{}".format(root_dir, object.replace(" ", "_")) for object in objects]
id2path = {i:p for i, p in enumerate(paths)}

print(f"generating images ...")
for id, prompt in enumerate(prompts):
    image = pipe(
        prompt, 
        num_inference_steps=config_dict["num_inference_steps"], 
        guidance_scale=config_dict["cfg"]).images[0]
    image.save(f"{id2path[id]}.png")