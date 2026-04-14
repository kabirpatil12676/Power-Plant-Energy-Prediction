import json

with open("e:/Power-Plant-Energy-Prediction/ANN_Regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    new_source = []
    for line in cell.get("source", []):
        # Replace the literal backslash `\` followed by `n` sequence with actual python newline characters
        new_source.append(line.replace("\\n", "\n"))
    cell["source"] = new_source

with open("e:/Power-Plant-Energy-Prediction/ANN_Regression.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
