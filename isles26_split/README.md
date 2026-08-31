# ISLES26 three-fold split

This directory records the site-aware, metadata-stratified three-fold split
used for the ISLES26 training and validation runs in this repository. It does
not contain medical images, lesion masks, or tabular source files.

## Manifest

`isles26_soopincluded_3fold_2026_08_09.json` contains 1,453 cases from 55
sites. Its top-level `training` array assigns every case to one of three folds:

| JSON fold | Cases | Repository configuration |
|---:|---:|---|
| 0 | 485 | fold 1 |
| 1 | 484 | fold 2 |
| 2 | 484 | fold 3 |

Image, label, and metadata paths in the manifest are relative to the dataset
root. Configure `environment.dataset.data_root` to the directory containing
the corresponding site directories, and configure
`environment.dataset.split_file` to this manifest's absolute path.

The manifest's SHA-256 digest is:

```text
8082E4574402E6BA932C716D0395AAB32107919E7EADB29B0F42626F823E2986
```

Access to the underlying ISLES26/ATLAS data remains subject to the dataset and
challenge terms. The data themselves are not redistributed here.
