import json
import os

output_dir = "outputs/hackathon_demo/experiment_20250607_225122"
original_path = os.path.join(output_dir, "class_to_idx.json")
new_path = os.path.join(output_dir, "class_to_idx_reencoded.json")

try:
    # Read the original file with utf-8-sig to handle BOMs
    with open(original_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    # Write to a new file with standard UTF-8 encoding (no BOM)
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully re-encoded {original_path} to {new_path}")
    print("You can now update app.py to point to 'class_to_idx_reencoded.json'.")
    
except Exception as e:
    print(f"Error re-encoding JSON: {e}") 