"""Safe amplitude normalization without distorting acoustic features."""

import numpy as np


class AudioNormalizer:
    """Performs safe, non-destructive amplitude normalization on float32 PCM audio."""

    @staticmethod
    def normalize(samples: np.ndarray) -> np.ndarray:
        """
        Normalize audio samples safely into [-1.0, 1.0].

        - If peak exceeds 1.0, scale down to prevent distortion.
        - Preserves natural dynamic range without aggressively amplifying noise.
        - Clamps any extreme numerical outliers to [-1.0, 1.0].
        """
        if len(samples) == 0:
            return samples

        peak = float(np.max(np.abs(samples)))

        # If audio exceeds full scale, attenuate to 1.0
        if peak > 1.0:
            samples = samples / peak

        # Cleanly clip any boundary numerical imprecision
        samples = np.clip(samples, -1.0, 1.0)

        return samples.astype(np.float32)
