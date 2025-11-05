import os, random, shutil

ROOT = "data/ESC-50-master/audio"
OUT_ROOT = "data/esc50"
TRAIN_DIR = os.path.join(OUT_ROOT, "train")
TEST_DIR = os.path.join(OUT_ROOT, "test")
EXT = ".wav"
RATIO = 0.8
SEED = 1337
USE_COPY = False

def list_files(root, ext):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                out.append(os.path.join(r, f))
    return sorted(out)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def place(target_root, root, files):
    for src in files:
        rel = os.path.relpath(src, root)
        dst = os.path.join(target_root, rel)
        ensure_dir(os.path.dirname(dst))
        if os.path.lexists(dst):
            continue
        if USE_COPY:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)

def main():
    ensure_dir(TRAIN_DIR); ensure_dir(TEST_DIR)
    files = list_files(ROOT, EXT)
    random.seed(SEED); random.shuffle(files)
    k = int(len(files) * RATIO)
    train_files, test_files = files[:k], files[k:]
    place(TRAIN_DIR, ROOT, train_files)
    place(TEST_DIR, ROOT, test_files)
    print(f"total={len(files)} train={len(train_files)} test={len(test_files)}")
    print(f"train -> {TRAIN_DIR}")
    print(f"test  -> {TEST_DIR}")

if __name__ == "__main__":
    main()
