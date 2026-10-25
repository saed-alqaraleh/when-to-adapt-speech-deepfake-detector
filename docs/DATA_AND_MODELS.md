# Datasets and detector access

The following official sources describe the independent data and model dependencies. Dataset and model licences are determined by the respective publishers, not by this repository.

| Resource | Official or original source | Use |
| --- | --- | --- |
| ASVspoof 2021 DF | https://www.asvspoof.org/index2021.html | Deepfake evaluation trials and keys |
| ASVspoof 5 | https://github.com/asvspoof-challenge/asvspoof5 | Baseline and evaluation protocols |
| AASIST | https://github.com/clovaai/aasist | Source detector implementation and original weights |
| SSL anti-spoofing | https://github.com/TakHemlata/SSL_Anti-spoofing | SSL-AASIST reference implementation |
| XLSR-Mamba | https://huggingface.co/AustinXiao/XLSR-Mamba-DF | Detector checkpoint referenced by the experiment |

The result archive includes only numeric evaluation records and protocol descriptors. For a new end-to-end run, create a local data directory outside the repository, obtain any necessary permissions, and use the entry-point modules described in the source files. Absolute paths to private working machines and hosted notebooks are not part of the release.
