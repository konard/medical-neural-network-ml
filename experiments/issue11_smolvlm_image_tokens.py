"""Count the image tokens SmolVLM's processor inserts for one image (issue #9).

Run offline from the Hugging Face cache with ``python -I``. It prints the image
size, whether image splitting was on, the prompt length and the image tokens.
On 2026-10-16 it printed 1 137 / 1 088 with splitting and 77 / 64 without, for
both 384 and 512 px.
"""

from PIL import Image
from transformers import AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"

processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe."}],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
for size in (384, 512):
    image = Image.new("RGB", (size, size), (90, 90, 90))
    for split in (True, False):
        batch = processor(
            text=[prompt],
            images=[[image]],
            return_tensors="pt",
            do_image_splitting=split,
        )
        ids = batch["input_ids"][0]
        print(size, split, ids.shape[0], int((ids == image_token).sum()))
