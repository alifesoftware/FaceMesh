// Create Virtual Python Environment
python3 -m venv .venv

// Activate Virtual Python Environment
source .venv/bin/activate

// Install Required Python Packages
pip install -r /Users/gifty/Development/FaceMesh/tools/requirements.txt

// Runs the BlazeFace Similar Face Script
python tools/reference_embed.py /Users/gifty/Development/FaceMesh/tools/dataset/virat01.jpg /Users/gifty/Development/FaceMesh/tools/dataset/virat02.jpg /Users/gifty/Development/FaceMesh/tools/dataset/srk00.jpg

// Runs End-to-End Pipeline Script - Tests Face Detection, Embedding Creation, and Clustering. 
// Best way to test is keep photos of same person in a folder, have 2-3 folders, and check the resulting clusters for a match
// Note: You can add any number of folders with --folder tag
python tools/reference_pipeline.py --folder /Users/gifty/Downloads/anuj --folder /Users/gifty/Downloads/kary --out ./clusters

// Other examples: Add EPS or add for Help
python tools/reference_pipeline.py --folder /Users/gifty/Downloads/anuj --folder /Users/gifty/Downloads/kary --eps 0.50 --out ./clusters (or instead of --out, add --quiet   # sweep without big output dirs)

// Also try this flag:
--dedupe-by-content to compare duplicate files across folders

// Help
python tools/reference_pipeline.py --help   # tuning block at bottom