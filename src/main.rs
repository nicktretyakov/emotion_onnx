//! Emotion recognition CLI (pure std version).
//! Ready to be extended with `ort` for real ONNX inference.

mod models;
mod preprocess;
mod fusion;

use std::env;
use std::path::PathBuf;
use std::process;

use crate::models::{EmotionModel, Modality};
use crate::preprocess::EmotionLabels;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(1);
    }

    match args[1].as_str() {
        "labels" => {
            println!("Supported emotion labels:");
            for (i, label) in EmotionLabels::all().iter().enumerate() {
                println!("  {}: {}", i, label);
            }
        }
        "face" => {
            let image = get_arg(&args, "--image").or_else(|| get_arg(&args, "-i"));
            let model = get_arg(&args, "--model")
                .or_else(|| get_arg(&args, "-m"))
                .unwrap_or_else(|| "models/face_emotion.onnx".to_string());
            let size: u32 = get_arg(&args, "--size")
                .and_then(|s| s.parse().ok())
                .unwrap_or(224);

            match image {
                Some(img) => {
                    if let Err(e) = run_face(&PathBuf::from(img), &PathBuf::from(model), size) {
                        eprintln!("Error: {}", e);
                        process::exit(1);
                    }
                }
                None => {
                    eprintln!("Error: --image <path> is required");
                    process::exit(1);
                }
            }
        }
        "multimodal" => {
            let image = get_arg(&args, "--image").map(PathBuf::from);
            let text = get_arg(&args, "--text");
            if let Err(e) = run_multimodal(image, text) {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_help();
            process::exit(1);
        }
    }
}

fn get_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_help() {
    println!(
        r#"emotion_onnx — multimodal emotion recognition

USAGE:
    emotion_onnx <COMMAND> [OPTIONS]

COMMANDS:
    labels                  List supported emotion labels
    face --image <path>     Recognize emotion from face image
    multimodal              Late fusion of available modalities
    help                    Show this help

OPTIONS:
    --image, -i <path>      Path to image
    --model, -m <path>      Path to ONNX model (default: models/face_emotion.onnx)
    --size <n>              Input size (default: 224)
    --text <string>         Text for multimodal mode

EXAMPLES:
    emotion_onnx labels
    emotion_onnx face --image photo.jpg
    emotion_onnx multimodal --image photo.jpg --text "I am happy"
"#
    );
}

fn run_face(image_path: &PathBuf, model_path: &PathBuf, size: u32) -> Result<(), String> {
    println!("Loading model from {:?} ...", model_path);

    let mut model = EmotionModel::load(model_path, Modality::Face)?;
    println!("Preprocessing image {:?} (size={})", image_path, size);

    let input = preprocess::preprocess_image(image_path, size)?;
    let output = model.infer(&input)?;

    let labels = EmotionLabels::all();
    let (best_idx, best_score) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("\n=== Facial Emotion Recognition ===");
    println!("Image: {:?}", image_path);
    for (i, (label, score)) in labels.iter().zip(output.iter()).enumerate() {
        let marker = if i == best_idx { " <--" } else { "" };
        println!("  {:>10}: {:.4}{}", label, score, marker);
    }
    println!(
        "\nPredicted: {} ({:.2}%)",
        labels[best_idx],
        best_score * 100.0
    );
    Ok(())
}

fn run_multimodal(image: Option<PathBuf>, text: Option<String>) -> Result<(), String> {
    let mut face_probs = None;
    let mut text_probs = None;

    if let Some(img) = image {
        let model_path = PathBuf::from("models/face_emotion.onnx");
        if model_path.exists() {
            let mut model = EmotionModel::load(&model_path, Modality::Face)?;
            let input = preprocess::preprocess_image(&img, 224)?;
            face_probs = Some(model.infer(&input)?);
        } else {
            // mock even without model file
            face_probs = Some(vec![0.05, 0.03, 0.07, 0.55, 0.10, 0.08, 0.12]);
        }
    }

    if text.is_some() {
        // very simple mock based on keywords
        let t = text.as_ref().unwrap().to_lowercase();
        text_probs = Some(if t.contains("happy") || t.contains("рад") || t.contains("счастлив") {
            vec![0.05, 0.02, 0.03, 0.70, 0.05, 0.05, 0.10]
        } else if t.contains("sad") || t.contains("груст") {
            vec![0.05, 0.02, 0.05, 0.05, 0.70, 0.03, 0.10]
        } else if t.contains("angry") || t.contains("зл") || t.contains("angry") {
            vec![0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        } else {
            vec![0.10, 0.05, 0.10, 0.15, 0.10, 0.10, 0.40]
        });
    }

    let fused = fusion::late_fusion(face_probs, None, text_probs, &[0.6, 0.0, 0.4]);
    let labels = EmotionLabels::all();
    let (best_idx, best_score) = fused
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("\n=== Multimodal Emotion Recognition (Late Fusion) ===");
    for (i, (label, score)) in labels.iter().zip(fused.iter()).enumerate() {
        let marker = if i == best_idx { " <--" } else { "" };
        println!("  {:>10}: {:.4}{}", label, score, marker);
    }
    println!(
        "\nPredicted: {} ({:.2}%)",
        labels[best_idx],
        best_score * 100.0
    );
    Ok(())
}
