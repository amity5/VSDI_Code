from pathlib import Path
import numpy as np
import scipy.io as sio


class MatReader:
    """
    Handles loading condition arrays from MATLAB .mat files.
    """

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def load_mat(self, mat_path: Path) -> dict:
        full_path = self._resolve_path(mat_path)
        return sio.loadmat(full_path)

    def load_condition(self, mat_path: Path, condition_key: str) -> np.ndarray:
        """
        Load specific condition variable from MAT file.

        Returns:
            ndarray of shape (n_pixels, n_frames)
        """
        mat = self.load_mat(mat_path)

        if condition_key not in mat:
            raise KeyError(
                f"Condition '{condition_key}' not found in {mat_path}"
            )

        arr = mat[condition_key]

        if not isinstance(arr, np.ndarray):
            raise ValueError("Loaded condition is not a numpy array")

        return arr

    def list_variables(self, mat_path: Path):
        mat = self.load_mat(mat_path)
        return list(mat.keys())

    def _resolve_path(self, mat_path: Path) -> Path:
        """
        Resolve relative paths against data_root.
        """
        mat_path = Path(mat_path)
        if mat_path.is_absolute():
            return mat_path
        return self.data_root / mat_path
