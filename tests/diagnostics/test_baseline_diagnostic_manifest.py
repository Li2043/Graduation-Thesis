"""Manifest path rules."""

from thesis.diagnostics.stage7a0_manifest import is_absolute_path_string


def test_manifest_rejects_windows_absolute_paths():
    assert is_absolute_path_string(r"C:\tmp\a.csv")
    assert not is_absolute_path_string("output/endpoint_tables/a.csv")
