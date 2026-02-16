"""Dataset-specific file/entry indexing and loading adapters."""

from whirdetective.data.adapters.cwru import (
    infer_cwru_label_from_path,
    list_cwru_mat_files,
    load_cwru_channels,
)
from whirdetective.data.adapters.paderborn import list_paderborn_mat_entries
from whirdetective.data.adapters.paderborn import (
    infer_paderborn_label_from_archive,
    iter_paderborn_mat_payloads,
    list_paderborn_archives,
    load_paderborn_channels_from_mat_payload,
)
from whirdetective.data.adapters.xjtu_sy import XjtuCsvEntry, iter_xjtu_csv_payloads, list_xjtu_parts

__all__ = [
    "infer_cwru_label_from_path",
    "infer_paderborn_label_from_archive",
    "iter_paderborn_mat_payloads",
    "iter_xjtu_csv_payloads",
    "list_cwru_mat_files",
    "list_paderborn_archives",
    "list_xjtu_parts",
    "load_cwru_channels",
    "load_paderborn_channels_from_mat_payload",
    "list_paderborn_mat_entries",
    "XjtuCsvEntry",
]
