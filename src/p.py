from pathlib import Path

print("train:", len(list(Path("data/miditok_out/data_chunks/train").glob("**/*.mid"))))
print("valid:", len(list(Path("data/miditok_out/data_chunks/valid").glob("**/*.mid"))))
print("test :", len(list(Path("data/miditok_out/data_chunks/test").glob("**/*.mid"))))