import shutil
import os

target = "data/chroma_db"
if os.path.exists(target):
    shutil.rmtree(target)
    print(f"✓ Deleted {target}")
else:
    print(f"  {target} already gone")

for f in ["data/bm25_index.pkl", "data/chunks.pkl"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ Deleted {f}")
    else:
        print(f"  {f} already gone")

print("\ndata/ folder now contains:")
for item in os.listdir("data"):
    print(f"  {item}")

print("\n✓ Ready to rebuild")