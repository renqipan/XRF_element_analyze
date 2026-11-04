# Additional raw MCA spectra

This directory originally contained 20 raw sample spectra added while version 0.2 was being published; 17 are currently present. Analysis uses files actually present and does not restore removed samples. Original bytes are retained. Filenames describe different nominal acquisition conditions and are not parsed as verified tube-voltage metadata.

Match each sample to its appropriate calibration and common-scale sensitivity before quantitative batch analysis. The original raw files in this directory have not been modified or treated as a shared calibrated dataset.

The user confirmed that `data/backgroud_spectrum.mca` is a pure-Pb XRF reference spectrum. It is byte-for-byte identical to `Pb_50keV_20uA_back3mm.mca` in this directory, not a generic independent blank. See the project-root `README.md` and `data/README.md` for usage and interpretation limits.
