data_analysis = """
Median image size: (512, 512, 128)
Voxel spacing (median): (1.0, 1.0, 1.0)
Modality: CT
Organ: Liver
Num samples: 131
Min intensity: -100
Max intensity: 400
"""

with open("/content/drive/MyDrive/TesisSegMamba/data_analysis_result.txt", "w") as f:
    f.write(data_analysis.strip())

print("✅ Archivo 'data_analysis_result.txt' generado.")
