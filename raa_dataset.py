import json
from torch.utils.data import Dataset
from datetime import datetime

class PatientDataset(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path (str): Path to the JSON file.
            transform (callable, optional): A function/transform that takes
                the segments list and returns a processed version.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = parse_timestamps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        patient_id = entry['patient_id']
        segments = entry['segments']
        
        if self.transform:
            segments = self.transform(segments)
        
        return {
            'patient_id': patient_id,
            'segments': segments
        }


def parse_timestamps(segments):
    for seg in segments:
        seg['timestamp'] = datetime.fromisoformat(seg['timestamp'])
    return segments

def patient_collate(batch):
    # batch is a list of 1 element; just return that element
    return batch[0]