/// Late fusion: weighted average of probability vectors.
pub fn late_fusion(
    face: Option<Vec<f32>>,
    audio: Option<Vec<f32>>,
    text: Option<Vec<f32>>,
    weights: &[f32],
) -> Vec<f32> {
    let n_classes = 7;
    let mut fused = vec![0.0f32; n_classes];
    let mut total_weight = 0.0f32;

    let modalities = [
        (face.as_ref(), weights.get(0).copied().unwrap_or(0.5)),
        (audio.as_ref(), weights.get(1).copied().unwrap_or(0.3)),
        (text.as_ref(), weights.get(2).copied().unwrap_or(0.2)),
    ];

    for (probs, w) in modalities {
        if let Some(p) = probs {
            if p.len() == n_classes {
                for i in 0..n_classes {
                    fused[i] += p[i] * w;
                }
                total_weight += w;
            }
        }
    }

    if total_weight > 0.0 {
        for v in &mut fused {
            *v /= total_weight;
        }
    } else {
        let u = 1.0 / n_classes as f32;
        fused.fill(u);
    }
    fused
}
