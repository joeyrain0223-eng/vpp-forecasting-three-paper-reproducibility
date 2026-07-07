# Code Availability

The package includes the Python scripts used to prepare the public OPSD data, run baselines, execute the GraphPatch experiments, generate figures, rebuild manuscript support files, and verify the package manifest.

For local manuscript builds, the original scripts are preserved with absolute workspace anchors. Before a public repository release, convert `BASE` and `ROOT` constants to repository-relative paths, then rerun the command sequence from a clean checkout:

```bash
bash RUN_COMMANDS.sh
python scripts/verify_paper1_supplement_package.py
```

The repository release should keep all non-public Hunan/Shandong files outside version control and outside any DOI-backed archive.
