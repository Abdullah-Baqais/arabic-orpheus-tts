import torch
import soundfile as sf
from unsloth import FastLanguageModel
from snac import SNAC

MODEL_NAME = "Lokatsu/orpheus-arabic-tts-16bit"
OUTPUT_WAV = "output.wav"
TARGET_SR = 24000

# Orpheus / SNAC token settings
AUDIO_TOKEN_OFFSET = 128266
END_OF_AUDIO_TOKEN = 128258  # stop if generated
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Helpers
# -------------------------
def extract_audio_token_ids(generated_ids, input_len):
    """
    Take only newly generated tokens, then keep audio tokens.
    """
    new_tokens = generated_ids[0][input_len:].tolist()

    audio_tokens = []
    for token in new_tokens:
        if token == END_OF_AUDIO_TOKEN:
            break
        if token >= AUDIO_TOKEN_OFFSET:
            audio_tokens.append(token - AUDIO_TOKEN_OFFSET)

    # SNAC/Orpheus decoding expects groups of 7
    usable_len = (len(audio_tokens) // 7) * 7
    audio_tokens = audio_tokens[:usable_len]

    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens were found in the generated output.")

    return audio_tokens


def redistribute_codes(code_list, device):
    """
    Convert flattened Orpheus audio token stream into 3 SNAC code layers.
    Pattern used by common Orpheus inference examples.
    """
    layer_1 = []
    layer_2 = []
    layer_3 = []

    n_frames = len(code_list) // 7
    for i in range(n_frames):
        base = 7 * i

        c0 = code_list[base]
        c1 = code_list[base + 1] - 4096
        c2 = code_list[base + 2] - (2 * 4096)
        c3 = code_list[base + 3] - (3 * 4096)
        c4 = code_list[base + 4] - (4 * 4096)
        c5 = code_list[base + 5] - (5 * 4096)
        c6 = code_list[base + 6] - (6 * 4096)

        # clamp to valid SNAC codebook range
        c0 = max(0, min(c0, 4095))
        c1 = max(0, min(c1, 4095))
        c2 = max(0, min(c2, 4095))
        c3 = max(0, min(c3, 4095))
        c4 = max(0, min(c4, 4095))
        c5 = max(0, min(c5, 4095))
        c6 = max(0, min(c6, 4095))

        layer_1.append(c0)

        layer_2.append(c1)
        layer_2.append(c4)

        layer_3.append(c2)
        layer_3.append(c3)
        layer_3.append(c5)
        layer_3.append(c6)

    codes = [
        torch.tensor(layer_1, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.long, device=device).unsqueeze(0),
    ]
    return codes


# -------------------------
# Load model
# -------------------------
print(f"Loading model on {DEVICE} ...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    load_in_4bit=False,   # merged 16-bit model
)

model = model.to(DEVICE)
FastLanguageModel.for_inference(model)

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE).eval()

# -------------------------
# Inference text
# -------------------------
text = "مَرْحَبًا بِكُمْ فِي هَذَا النِّظَامِ الصَّوْتِيِّ الجَدِيدِ."

# Some Orpheus checkpoints work better with a speaker-style prefix like "Speaker 0: "
prompt = text

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2200,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print("Generated shape:", outputs.shape)

# -------------------------
# Convert generated tokens -> audio
# -------------------------
audio_token_ids = extract_audio_token_ids(outputs, input_len)
print("Number of audio tokens:", len(audio_token_ids))

codes = redistribute_codes(audio_token_ids, DEVICE)

with torch.inference_mode():
    audio_hat = snac_model.decode(codes)

audio = audio_hat.squeeze().detach().float().cpu().numpy()
sf.write(OUTPUT_WAV, audio, TARGET_SR)

print(f"Saved audio to: {OUTPUT_WAV}")
