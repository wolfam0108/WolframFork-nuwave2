# NU-Wave2 — Official PyTorch Implementation

**NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates**<br>
Seungu Han, Junhyeok Lee @ [MINDsLab Inc.](https://github.com/mindslab-ai), SNU

[![arXiv](https://img.shields.io/badge/arXiv-2206.08545-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2206.08545) [![GitHub Repo stars](https://img.shields.io/github/stars/mindslab-ai/nuwave2?color=yellow&label=NU-Wave2&logo=github&style=flat-square)](https://github.com/mindslab-ai/nuwave2) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://mindslab-ai.github.io/nuwave2/)

> **Fork:** WolframFork-nuwave2 — добавлена чанковая обработка, стерео поддержка, совместимость с PyTorch 2.x

---

## 🚀 Quick Start (Inference)

### 1. Установка

```bash
# Клонировать репозиторий
git clone https://github.com/wolfam0108/WolframFork-nuwave2.git
cd WolframFork-nuwave2

# Создать conda окружение
conda create -n nuwave2 python=3.9 -y
conda activate nuwave2

# Установить зависимости
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install librosa==0.9.2 omegaconf pytorch_lightning scipy matplotlib numpy gdown

# Скачать pre-trained модель (48kHz)
gdown "https://drive.google.com/uc?id=11t0cQYx6ZadKQjmfGnqxUUH2UEk5Yzk7" -O nuwave2_48khz.ckpt
```

### 2. Запуск (чанковая обработка)

```bash
conda activate nuwave2
python inference_chunked.py -c nuwave2_48khz.ckpt -i "input.wav" --sr 22000 --gt --device cuda
```

### 3. Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `-c` | *обязателен* | Путь к чекпоинту |
| `-i` | *обязателен* | Путь к входному WAV файлу |
| `--sr` | *обязателен* | Эффективный sample rate входного аудио |
| `--gt` | `false` | Входной файл уже 48kHz (симуляция low-res) |
| `--device` | `cuda` | Устройство: cuda или cpu |
| `--chunk_sec` | `10.0` | Размер чанка в секундах |
| `--overlap_sec` | `2.0` | Перекрытие чанков в секундах |
| `-o` | auto | Путь к выходному файлу |
| `--mono` | `false` | Принудительно моно обработка |

### 4. Примеры

```bash
# Апсемплинг низкого разрешения (16kHz → 48kHz)
python inference_chunked.py -c nuwave2_48khz.ckpt -i low_quality.wav --sr 16000

# Восстановление сжатого 48kHz аудио (симуляция low-res)  
python inference_chunked.py -c nuwave2_48khz.ckpt -i compressed.wav --sr 22000 --gt

# Явный выходной файл
python inference_chunked.py -c nuwave2_48khz.ckpt -i input.wav --sr 22000 --gt -o output.wav
```

---

## 🔧 Доработки в этом форке

1. **Чанковая обработка** (`inference_chunked.py`) — overlap-add метод для длинных файлов без OOM
2. **Стерео поддержка** — обработка каждого канала отдельно, сохранение как стерео
3. **PyTorch 2.x совместимость** — исправлены `torch.stft`/`torch.istft` в `model.py` и `utils/stft.py`
4. **Автоскачивание модели** — через gdown из Google Drive

---

## Checkpoints

| Модель | Выход | Ссылка |
|--------|-------|--------|
| **48kHz модель** | 48 kHz | [Скачать](https://drive.google.com/file/d/11t0cQYx6ZadKQjmfGnqxUUH2UEk5Yzk7/view) |
| **16kHz модель** | 16 kHz | [Скачать](https://drive.google.com/file/d/1IZihqb0LKHLtqRjyhHBGxXHJhUwskVRo/view) |

---

## References
- [official NU-Wave pytorch implementation](https://github.com/mindslab-ai/nuwave)
- [ivanvovk's WaveGrad pytorch implementation](https://github.com/ivanvovk/WaveGrad)
- [lmnt-com's DiffWave pytorch implementation](https://github.com/lmnt-com/diffwave)

## Citation
```bib
@article{han2022nu,
  title={NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates},
  author={Han, Seungu and Lee, Junhyeok},
  journal={arXiv preprint arXiv:2206.08545},
  year={2022}
}
```

## Contact
If you have a question or any kind of inquiries, please contact Seungu Han at [hansw032@snu.ac.kr](mailto:hansw0326@snu.ac.kr)
