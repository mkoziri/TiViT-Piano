"""Purpose:
    Expose music-theory helpers such as the key-aware prior used when decoding
    model logits.

Key Functions/Classes:
    - KeyAwarePrior: Applies key estimation and prior rescoring to pitch-class
      logits.
    - build_key_profiles(): Returns normalized Krumhansl--Schmuckler key
      profiles for major and minor modes.

CLI:
    None.  Import this module from other code to access theory helpers.
"""

from .key_prior import KeyAwarePrior, KeyPriorConfig, build_key_profiles

__all__ = ["KeyAwarePrior", "KeyPriorConfig", "build_key_profiles"]
