# emotion_onnx

Rust-проект для мультимодального распознавания эмоций.

Текущая версия работает на **чистом std** (без внешних зависимостей) и использует mock-инференс.  
Готова к подключению настоящего ONNX Runtime (`ort`).

## Структура

```
emotion_onnx/
├── Cargo.toml
├── README.md
├── models/           # сюда кладите .onnx модели
│   └── README.md
├── data/             # тестовые изображения
├── src/
│   ├── main.rs       # CLI
│   ├── models.rs     # загрузка модели + infer()
│   ├── preprocess.rs # предобработка изображений
│   └── fusion.rs     # late fusion
```

## Сборка и запуск

```bash
cargo build --release

# Список эмоций
cargo run --release -- labels

# Распознавание по лицу (mock)
cargo run --release -- face --image data/test.jpg

# Мультимодальный режим
cargo run --release -- multimodal --image data/test.jpg --text "я очень злой"
cargo run --release -- multimodal --text "I am so happy today!"
```

## Поддерживаемые эмоции

```
0: angry
1: disgust
2: fear
3: happy
4: sad
5: surprise
6: neutral
```

## Как добавить настоящий ONNX

1. Добавьте в `Cargo.toml`:

```toml
[dependencies]
ort = { version = "2", features = ["download-binaries", "ndarray"] }
image = "0.25"
ndarray = "0.16"
```

2. Замените реализацию `EmotionModel` в `src/models.rs` на версию с `ort::Session`.
3. Реализуйте настоящую предобработку изображения в `src/preprocess.rs`.

## Лицензия

MIT
