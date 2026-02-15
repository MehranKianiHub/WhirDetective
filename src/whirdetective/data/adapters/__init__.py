"""Dataset-specific file/entry indexing and loading adapters."""

from whirdetective.data.adapters.cwru import (
    infer_cwru_label_from_path,
    list_cwru_mat_files,
    load_cwru_channels,
)
from whirdetective.data.adapters.paderborn import list_paderborn_mat_entries

__all__ = [
    "infer_cwru_label_from_path",
    "list_cwru_mat_files",
    "load_cwru_channels",
    "list_paderborn_mat_entries",
]
