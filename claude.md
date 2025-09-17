This project will explore the Global Flood Monitoring (GFM) data set available from: https://services.eodc.eu/browser/#/v1/collections/GFM

## Proof of concept goal

1. choose a regional AOI in E. Africa that would cover multiple overlapping/adjacent sentinel tiles. We want 3-4 adjacent/overlapping tiles in the geography/aoi not hundreds for this proof of concept.
2. Get the entire historical record of GFM images in that AOI. At each daily timestep composite them together 
to get the latest composite GFM flood extent for the AOI.


## Setup Instructions

The project uses Python 3.11.4 with pyenv for version and environment management.

1. **Environment Setup** (already configured):
   - Virtual environment: `ds-flood-gfm` 
   - Python version: 3.11.4 (set via `.python-version`)
   - Editable package installation with `pip install -e .`

2. **Jupyter Kernel**:
   - Kernel name: `ds-flood-gfm`
   - Display name: "Python (ds-flood-gfm)"

3. **Project Structure**:
   - `src/` directory for modules and packages
   - `data/` directory for all datasets (git-ignored)
     - `data/gfm/` for flood extent data
     - `data/ghsl/` for population data
   - `experiments/` directory for Claude Code experiments and analysis
     - `experiments/claude-tests/` for reusable experiment scripts and analysis
     - `experiments/temp-analysis/` for temporary/throwaway analysis files (likely to be deleted)
   - Build modular architecture as needed

## Python Libraries

- `pystac-client`: Connect to STAC API services
- `jupyter`: Notebook environment
- `ipykernel`: Jupyter kernel support
- dont put claude in commit messages


## Common pitfalls


- do not follow bad practices
Always follow best practices for custom module imports, NEVER do stuff like:
```
sys.path.append('../src')
```

Also adding a `.` before imports is not correct:

```
from .gfm_downloader import GFMDownloader

```