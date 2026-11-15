# docs

| File | What it is |
|---|---|
| [research.md](research.md) | Initial research: the executable, Switch 2 hardware, every source used with its license, NVN2 against NVN1, shaders |
| [performance.md](performance.md) | Frame times measured on real machines, with their date; users add theirs through the performance report form |
| [upscaling.md](upscaling.md) | DLSS, FSR 3.1, DLSS Frame Generation, FSR 3.1 frame generation and NVIDIA Reflex: settings and how they hook into the game |
| `logo.svg` | The repo's logo, the same chequered flag as `mkw.exe`'s icon, drawn by `tools/build/icon.py docs/logo.svg` |
| [i18n/](i18n/README.md) | The README in other languages (Simplified Chinese, Spanish, Arabic, Hindi, Brazilian and European Portuguese), built by `tools/docs/i18n_readme.py` |
| [images/](images/README.md) | Screenshots of the start screen for the README |
| [roadmap.md](roadmap.md) | The open work in order of priority, with the measurements behind it: the serial race frame first |
| `release-template.md` | What every release's notes repeat (install, updating, known issues, reporting); `tools/build/notes.py` fills it with the version's section of `CHANGELOG.md` |
