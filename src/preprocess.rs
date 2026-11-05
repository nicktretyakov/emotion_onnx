use std::path::Path;

pub struct EmotionLabels;

impl EmotionLabels {
    pub fn all() -> &'static [&'static str] {
        &["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    }
}

/// Mock preprocessing.
/// In real version: load image → resize → normalize → flatten or NCHW tensor.
pub fn preprocess_image(path: &Path, size: u32) -> Result<Vec<f32>, String> {
    if !path.exists() {
        return Err(format!("Image not found: {:?}", path));
    }
    // Return a dummy tensor of correct conceptual size
    // Real implementation would use the `image` crate:
    // let img = image::open(path)?;
    // let resized = image::imageops::resize(...);
    // convert to [1, 3, H, W] float32
    let dummy = vec![0.5f32; (3 * size * size) as usize];
    Ok(dummy)
}
