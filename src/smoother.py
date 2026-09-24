"""
Gesture probability temporal smoothing module.
"""

class GestureSmoother:
    """
    Stabilkan output probabilitas model ML menggunakan Exponential Moving Average (EMA)
    dan voter kestabilan n-frame untuk mencegah flickering antar-frame.
    """

    def __init__(self, classes, alpha=0.30, min_confidence=0.60, stable_frames=4):
        self.classes = classes
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.ema_probabilities = None
        self.candidate = None
        self.candidate_frames = 0
        self.stable_label = None

    def update(self, probabilities):
        """
        Update probabilitas frame baru dan kembalikan label stabil, kandidat, serta tingkat confidence.
        """
        probabilities = probabilities[0]
        if self.ema_probabilities is None:
            self.ema_probabilities = probabilities.copy()
        else:
            self.ema_probabilities = (
                self.alpha * probabilities
                + (1.0 - self.alpha) * self.ema_probabilities
            )

        index = int(self.ema_probabilities.argmax())
        label = self.classes[index]
        confidence = float(self.ema_probabilities[index])

        if label == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = label
            self.candidate_frames = 1

        if confidence >= self.min_confidence and self.candidate_frames >= self.stable_frames:
            self.stable_label = label

        return self.stable_label, label, confidence

    def reset(self):
        """Reset state saat tangan keluar dari frame kamera."""
        self.ema_probabilities = None
        self.candidate = None
        self.candidate_frames = 0
        self.stable_label = None

