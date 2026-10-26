# Wake word: "Hey Bi"

There is no pretrained "Hey Bi" model. Bi ships with openWakeWord's built-in **`hey_jarvis`** model as a
placeholder, so out of the box you say **"Hey Jarvis"** to wake it. Everything about the wake word is in
`config.yaml`:

```yaml
wake:
  engine: openwakeword     # openwakeword | porcupine
  model: hey_jarvis        # built-in name, or a path like models/hey_bi.onnx
  threshold: 0.5           # raise if it wakes by itself, lower if it misses you
  cooldown_s: 1.5
```

When the wake word fires, Bi plays a short rising chime (`audio.chime: true`) and starts listening.

## Option A: train a custom `hey_bi.onnx` with openWakeWord (free, fully local)

openWakeWord models are trained only on **synthetic speech**, so you don't need hundreds of recordings.

1. Open the official training notebook (Google Colab, free GPU tier is enough):
   <https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb>
2. Set the target phrase. "Hey Bi" is short; give the generator spelling variants that sound right, e.g.
   `target_phrase = ["hey bee", "hey bi"]`. Short two-syllable phrases false-trigger more easily, so
   consider **"Hey Bi-Bi"** or **"Okay Bi"** if you get too many false wakes.
3. Use the notebook defaults (about 30k–50k synthetic positives, ~1 hour on a Colab GPU). Add your own
   negative audio (TV, music, your typical room noise) if you have it.
4. Download the resulting `hey_bee.onnx` (or similar), save it as `models/hey_bi.onnx`, and set:
   ```yaml
   wake:
     model: models/hey_bi.onnx
   ```
5. Tune `threshold`: run `python main.py`, watch for false wakes over a normal hour, and nudge the value
   up (fewer false wakes) or down (more sensitive) in steps of 0.05.

Tips: record 20–50 real "Hey Bi" clips with your laptop mic and use the notebook's verifier-model step,
which makes the model much more reliable for *your* voice. Custom models live in `models/` (git-ignored).

## Option B: Picovoice Porcupine (very accurate; free personal key)

1. Create a free account at <https://console.picovoice.ai/> and copy your AccessKey into `.env`:
   `PICOVOICE_ACCESS_KEY=...`
2. In the console, train the keyword "Hey Bi" for your platform (Windows / macOS / Linux) and download
   the `.ppn` file into `models/`.
3. `pip install -r requirements-optional.txt` (installs `pvporcupine`), then:
   ```yaml
   wake:
     engine: porcupine
     porcupine_keyword_path: models/Hey-Bi_en_mac_v3_0_0.ppn
     threshold: 0.5     # used as Porcupine sensitivity
   ```
Without a `.ppn` file the Porcupine engine falls back to its built-in "jarvis" keyword.
Note: `.ppn` files are platform-specific and the free tier needs an internet check at start-up.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Never wakes | Lower `threshold` to 0.35; check `python -m bi.doctor` shows the right input device |
| Wakes by itself | Raise `threshold` to 0.6–0.7; train with more negative data |
| Wakes from Bi's own voice | Use headphones, or raise `threshold` |
| "models are missing" | `python -m bi.doctor --download` (needs internet once) |
