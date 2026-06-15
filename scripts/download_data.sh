#!/bin/bash
# Download Replogle K562/RPE1 data for CausalCellBench.
#
# Three options, in order of preference:
#
# Option 1: CausalBench package (automatic, but requires working figshare access)
#   python3 -c "from causalscbench.data_access.create_dataset import CreateDataset; \
#     c = CreateDataset('data/causalbench_cache', filter=True); c.load()"
#
# Option 2: Direct download via wget (works on HPC/Linux)
#   wget -O data/k562.h5ad "https://plus.figshare.com/ndownloader/files/35773219"
#   wget -O data/rpe1.h5ad "https://plus.figshare.com/ndownloader/files/35775606"
#
# Option 3: Manual browser download from GEO GSE221321
#   1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221321
#   2. Download the supplementary files
#   3. Place in data/ directory
#
# After download, verify:
#   python3 -c "import anndata; a = anndata.read_h5ad('data/k562.h5ad'); print(a.shape)"
#   Expected: (162751, ~8000)

set -e
mkdir -p data/causalbench_cache

echo "Attempting download via wget..."
wget --no-check-certificate -O data/causalbench_cache/k562.h5ad \
    "https://plus.figshare.com/ndownloader/files/35773219" || {
    echo "wget failed. Try Option 3 (manual download)."
    exit 1
}

echo "Downloading RPE1..."
wget --no-check-certificate -O data/causalbench_cache/rpe1.h5ad \
    "https://plus.figshare.com/ndownloader/files/35775606" || {
    echo "RPE1 download failed."
}

echo "Downloading summary stats..."
wget -O data/causalbench_cache/summary_stats.xlsx \
    "https://ars.els-cdn.com/content/image/1-s2.0-S0092867422005979-mmc2.xlsx"

echo "Downloads complete. Verify with:"
echo "  python3 -c \"import anndata; a = anndata.read_h5ad('data/causalbench_cache/k562.h5ad'); print(a.shape)\""
