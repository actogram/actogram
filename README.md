# actogram

## Find rest-activity intervals in wearable step counts data

Install in editable mode from the repo root:

```bash
pip install -e ".[dev]"
```

Then:

```python
from actogram import peakts, intervalts, plot
```

---

# Modules

`peakts` (peak time stamps) contains functions to extract circadian rhythm timestamps based on ACTIVITY DENSITY PEAK DETECTION.

`intervalts` (interval time stamps) contains functions to extract circadian rhythm timestamps based on REST INTERVALS DETECTION.

`checks` contains decorator functions for sanity checks of input parameters and data.

`plot` contains utility functions to visualize
analysis of circadian rhythm parameters.

---

Copyright © 2025 Actogram LTD (https://actogram.ai)
