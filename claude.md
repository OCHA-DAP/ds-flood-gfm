This project will explore the Global Flood Monitoring (GFM) data set available from: https://services.eodc.eu/browser/#/v1/collections/GFM

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
   - Use `src/` directory for modules and packages
   - Build modular architecture as needed

## Python Libraries

- `pystac-client`: Connect to STAC API services
- `jupyter`: Notebook environment
- `ipykernel`: Jupyter kernel support