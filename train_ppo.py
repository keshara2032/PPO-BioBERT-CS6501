from raa_dataset import PatientDataset, patient_collate
from torch.utils.data import DataLoader


json_path = './data/output.json'

dataset = PatientDataset(json_path)
print(f"Number of patients in the dataset: {len(dataset)}")


# pyTorch DataLoader ** Only batch size of 1 is supported
patinet_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=patient_collate)
for batch in patinet_loader:
    print("--" * 20)
    print(f"Batch patient ID: {batch['patient_id']}")
    print(f"Batch segments: {batch['segments']}")
    print("--" * 20)
    # break  # Just to show one batch