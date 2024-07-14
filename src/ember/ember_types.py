"""Type aliases for Ember.

Many cv2 functions operate on numpy arrays. This can make it hard to reason about code, so we define a few interpretable aliases here.
"""

import numpy as np

type Contour = np.ndarray
type Image = np.ndarray
