use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Modality {
    Face,
    Audio,
    Text,
}

pub struct EmotionModel {
    #[allow(dead_code)]
    modality: Modality,
}

impl EmotionModel {
    pub fn load(path: &Path, modality: Modality) -> Result<Self, String> {
        if path.exists() {
            println!(
                "[info] Found model file {:?} (mock inference — add `ort` crate for real ONNX)",
                path
            );
        } else {
            println!("[warn] Model file {:?} not found — using pure mock", path);
        }
        Ok(Self { modality })
    }

    /// Returns probability distribution over 7 emotions.
    /// Currently a deterministic mock biased toward "happy".
    /// Replace body with real ONNX Runtime call when ready.
    pub fn infer(&mut self, _input: &[f32]) -> Result<Vec<f32>, String> {
        // Mock: angry, disgust, fear, happy, sad, surprise, neutral
        Ok(vec![0.05, 0.03, 0.07, 0.55, 0.10, 0.08, 0.12])
    }
}
