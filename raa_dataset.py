import json
from torch.utils.data import Dataset
from datetime import datetime

class PatientDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): Path to the JSON file.
            transform (callable, optional): A function to apply to the segments list.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # parse timestamps by default
        self.transform = transform or parse_timestamps

        # build the global list of unique events
        events = set()
        for entry in self.data:
            for seg in entry['segments']:
                if seg['event_type'] == 'procedure' or seg['event_type'] == 'medication':
                    # only include procedures and medications
                    events.add(seg['event'])
        # optionally, only include procedures:
        # events = {seg['event']
        #           for entry in self.data
        #           for seg in entry['segments']
        #           if seg['event_type']=='procedure'}

        self.unique_events = sorted(events)
        print(f"Dataset: number of unique procedures and medications: {len(self.unique_events)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry     = self.data[idx]
        patient_id= entry['patient_id']
        segments  = entry['segments']

        # e.g. convert timestamp strings → datetime
        if self.transform:
            segments = self.transform(segments)

        return {
            'patient_id':   patient_id,
            'segments':     segments,
            'unique_events': self.unique_events
        }

def parse_timestamps(segments):
    for seg in segments:
        seg['timestamp'] = datetime.fromisoformat(seg['timestamp'])
    return segments

# simple “identity” collate (batch_size=1)
def single_item_collate(batch):
    # batch is a list of exactly one sample
    return batch[0]
