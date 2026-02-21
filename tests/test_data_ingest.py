from src.data_ingest import ingest_lfw

dataset, splits = ingest_lfw()

print("Total images:", len(dataset))
print("Train size:", len(splits["train"]))
print("Val size:", len(splits["val"]))
print("Test size:", len(splits["test"]))

print("First sample label:", dataset[0][0])
print("Image shape:", dataset[0][1].shape)
