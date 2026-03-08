import re
import json
import os
import torch
import soundfile as sf
import numpy as np
from bs4 import BeautifulSoup


def get_next_run_dir(base='output'):
    os.makedirs(base, exist_ok=True)
    existing = [
        int(d.replace('output', ''))
        for d in os.listdir(base)
        if d.startswith('output') and d.replace('output', '').isdigit()
    ]
    next_id = max(existing, default=0) + 1
    out_dir = os.path.join(base, f'output{next_id}')
    os.makedirs(out_dir)
    return out_dir


model = torch.package.PackageImporter('v5_ru.pt').load_pickle('tts_models', 'model')

with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
text = BeautifulSoup(text, 'html.parser').get_text()

sample_rate = 48000
speaker = 'eugene'
sentence_pause_sec = 0.3
paragraph_pause_sec = 0.8
sentence_pause = np.zeros(int(sample_rate * sentence_pause_sec))
paragraph_pause = np.zeros(int(sample_rate * paragraph_pause_sec))

paragraphs = text.strip().split('\n')
paragraphs = [p.strip() for p in paragraphs if p.strip()]

audio_chunks = []
timings = []
cursor_samples = 0

for paragraph in paragraphs:
    sentences = re.split(r'(?<=[.!?…])\s+', paragraph.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    for sentence in sentences:
        print(f"Generating: {sentence[:50]}...")
        audio = model.apply_tts(text=sentence, speaker=speaker, sample_rate=sample_rate)
        chunk = audio.numpy()
        start_sec = round(cursor_samples / sample_rate, 4)
        cursor_samples += len(chunk)
        end_sec = round(cursor_samples / sample_rate, 4)
        timings.append({
            "sentence": sentence,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })
        audio_chunks.append(chunk)
        audio_chunks.append(sentence_pause)
        cursor_samples += len(sentence_pause)
    audio_chunks.append(paragraph_pause)
    cursor_samples += len(paragraph_pause)

full_audio = np.concatenate(audio_chunks)

out_dir = get_next_run_dir()

sf.write(os.path.join(out_dir, 'audio.wav'), full_audio, sample_rate)
print(f"Done! Saved to {out_dir}/audio.wav")

with open(os.path.join(out_dir, 'timings.json'), 'w', encoding='utf-8') as f:
    json.dump(timings, f, ensure_ascii=False, indent=2)
print(f"Saved timings.json with {len(timings)} sentences")

with open(os.path.join(out_dir, 'text.txt'), 'w', encoding='utf-8') as f:
    f.write(text.strip())
print(f"Saved text.txt")

print(f"\nAll files saved to: {out_dir}/")