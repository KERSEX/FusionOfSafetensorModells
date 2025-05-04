from safetensors.torch import load_file, save_file

# Liste der Shards
shards = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

# Alle geladenen Tensors zusammenfügen (verlustfrei)
model_data = {}
all_keys = set()

for shard in shards:
    shard_data = load_file(shard)
    duplicate_keys = set(shard_data.keys()) & all_keys

    if duplicate_keys:
        raise ValueError(f"Doppelte Keys gefunden in {shard}: {duplicate_keys}")

    model_data.update(shard_data)
    all_keys.update(shard_data.keys())
    print(f"{shard} geladen mit {len(shard_data)} Tensors.")

# Zusammengeführte Datei speichern
output_file = "Mistral7B.safetensors"
save_file(model_data, output_file)
print(f"Zusammengeführtes Modell gespeichert als: {output_file}")
