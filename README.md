# Arabic Orpheus TTS Fine-tuning

Fine-tuning `unsloth/orpheus-3b-0.1-ft` for Arabic speech synthesis using a 2-stage pipeline:

1. Multi-speaker Arabic adaptation
2. Single-speaker specialization

## Highlights

- Formal Arabic / MSA-oriented output
- 2-stage LoRA fine-tuning
- Arabic text normalization
- SNAC-based audio tokenization
- Demo audio samples included

## Important note on data

The training dataset is **not included** in this repository.

I used licensed/restricted data and cannot redistribute:
- raw audio
- metadata
- transcriptions
- processed dataset artifacts derived from the source corpus

This repository only contains the training/inference code, configuration, and demo outputs.

## Repository contents

- `inference.py` — generate speech from text
- `demos/` — sample generated audio

## Training setup

- Base model: `unsloth/orpheus-3b-0.1-ft`
- Stage 1: Arabic multi-speaker training
- Stage 2: single-speaker fine-tuning on a female voice
- Hardware used: L40S
- LoRA fine-tuning with Unsloth

## 🔊 Demos

- **Any_help.wav** — [Open audio](./demos/Any_help.wav)
- **Greetings.wav** — [Open audio](./demos/Greetings.wav)
- **Greetings_2.wav** — [Open audio](./demos/Greetings_2.wav)

## Fine tuned model

If you want to try the model you will find it at:
`Lokatsu/orpheus-arabic-tts-16bit`
## Notes

This project is intended for research and educational purposes. Please respect the terms of any upstream model and dataset licenses.


## Citation

```
@misc{toyin2025arvoicemultispeakerdatasetarabic,
      title={ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis}, 
      author={Hawau Olamide Toyin and Rufael Marew and Humaid Alblooshi and Samar M. Magdy and Hanan Aldarmaki},
      year={2025},
      eprint={2505.20506},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20506}, 
}
```
