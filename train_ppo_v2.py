#!/usr/bin/env python3
"""
train_ppo_biobert.py

1. Loads PatientDataset (with unique_events)
2. Flattens out all ‘procedure’ and ‘medication’ segments into (text, label) pairs
3. Runs supervised cross‑entropy warm‑up
4. Fine‑tunes with PPO (using GAE, 0/1 reward)
5. Logs metrics and plots them at the end
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from raa_dataset import PatientDataset
import matplotlib.pyplot as plt

# ─── Hyperparameters ────────────────────────────────────────────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JSON_PATH         = "./data/output.json"

# Supervised pre‑training
PRETRAIN_EPOCHS   = 3
PRETRAIN_BSZ      = 16

# PPO / RL
LR                = 1e-5
GAMMA             = 0.9
LAMBDA            = 0.95
EPS_CLIP          = 0.1
K_EPOCHS          = 4
PPO_BATCH_SIZE    = 128
MAX_SEQ_LEN       = 128
TOTAL_PPO_UPDATES = 500
EVAL_INTERVAL     = 50
LOG_INTERVAL      = 20
# ────────────────────────────────────────────────────────────────────────────────

class PPOBioBERT(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.bert        = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        hidden            = self.bert.config.hidden_size
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        logits = self.policy_head(pooled)
        value  = self.value_head(pooled).squeeze(-1)
        return logits, value

def sample_and_tokenize(data, tokenizer, batch_size=1):
    texts, labels = zip(*random.sample(data, batch_size))
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN
    )
    return (
        enc.input_ids.to(DEVICE),
        enc.attention_mask.to(DEVICE),
        torch.tensor(labels, dtype=torch.long, device=DEVICE),
    )

def main():
    random.seed(42)
    torch.manual_seed(42)

    # 1) Load dataset & build action space
    ds         = PatientDataset(JSON_PATH)
    all_events = ds.unique_events
    event2idx  = {e:i for i,e in enumerate(all_events)}
    num_actions = len(all_events)
    print(f"Loaded {len(ds)} patients → {num_actions} unique events")

    # 2) Flatten segments
    event_data = []
    for entry in ds.data:
        for seg in entry["segments"]:
            if seg["event_type"] not in ("procedure","medication"):
                continue
            text  = seg["medic_note_segment"].strip() or seg["event"]
            label = event2idx[seg["event"]]
            event_data.append((text, label))
    print(f"→ {len(event_data)} samples")

    # 3) Setup model, tokenizer, optimizer, losses
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model     = PPOBioBERT(num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    # prepare logging containers
    logs = {
        "updates":        [],
        "avg_rewards":    [],
        "policy_losses":  [],
        "value_losses":   [],
        "ce_evals":       [],
        "ce_accs":        []
    }

    # 4) Supervised warm‑up
    print("\n[1] Supervised warm‑up")
    for epoch in range(1, PRETRAIN_EPOCHS+1):
        random.shuffle(event_data)
        total_loss, count = 0.0, 0
        for i in range(0, len(event_data), PRETRAIN_BSZ):
            batch = event_data[i:i+PRETRAIN_BSZ]
            ids, attn, labels = sample_and_tokenize(batch, tokenizer, len(batch))
            logits, _ = model(ids, attn)
            loss = ce_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(batch)
            count += len(batch)
        print(f"  Epoch {epoch}/{PRETRAIN_EPOCHS} — CE loss: {total_loss/count:.4f}")

    # 5) PPO fine‑tuning with GAE
    print("\n[2] PPO fine‑tuning")
    for update in range(1, TOTAL_PPO_UPDATES+1):
        mb_ids, mb_attn       = [], []
        mb_actions, mb_oldlp  = [], []
        mb_rewards, mb_values = [], []

        # collect episodes
        for _ in range(PPO_BATCH_SIZE):
            ids, attn, labels = sample_and_tokenize(event_data, tokenizer, 1)
            with torch.no_grad():
                logits, value = model(ids, attn)
                dist          = Categorical(logits=logits)
                action        = dist.sample().squeeze(0)
                old_logp      = dist.log_prob(action).squeeze(0)
            reward = 1.0 if action.item()==labels.item() else 0.0

            mb_ids.append(   ids.squeeze(0)    )
            mb_attn.append( attn.squeeze(0)   )
            mb_actions.append(action          )
            mb_oldlp.append(old_logp.detach() )
            mb_rewards.append(reward          )
            mb_values.append(value.item()     )

        # to tensors + append V(end)=0
        mb_ids     = torch.stack(mb_ids)
        mb_attn    = torch.stack(mb_attn)
        mb_actions = torch.stack(mb_actions)
        mb_oldlp   = torch.stack(mb_oldlp)
        mb_rewards = torch.tensor(mb_rewards, device=DEVICE)
        mb_values  = torch.tensor(mb_values+[0.0], device=DEVICE)

        # GAE advantages + returns
        gae         = 0.0
        advs, rets  = [], []
        for i in reversed(range(len(mb_rewards))):
            delta = mb_rewards[i] + GAMMA*mb_values[i+1] - mb_values[i]
            gae   = delta + GAMMA*LAMBDA*gae
            advs.insert(0, gae)
            rets.insert(0, gae+mb_values[i])
        advantages = torch.tensor(advs, device=DEVICE)
        returns    = torch.tensor(rets, device=DEVICE)
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)

        # PPO update
        for _ in range(K_EPOCHS):
            logits, values = model(mb_ids, mb_attn)
            dist           = Categorical(logits=logits)
            new_logp       = dist.log_prob(mb_actions)
            entropy        = dist.entropy().mean()

            ratio  = (new_logp - mb_oldlp).exp()
            s1     = ratio * advantages
            s2     = torch.clamp(ratio,1-EPS_CLIP,1+EPS_CLIP)*advantages
            p_loss = -torch.min(s1,s2).mean()
            v_loss = mse_loss(values, returns)
            loss   = p_loss + 0.5*v_loss - 0.01*entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log metrics
        if update % LOG_INTERVAL == 0:
            avg_r = mb_rewards.mean().item()
            logs["updates"].append(update)
            logs["avg_rewards"].append(avg_r)
            logs["policy_losses"].append(p_loss.item())
            logs["value_losses"].append(v_loss.item())
            print(f"[Update {update:4d}] avg_reward={avg_r:.3f} "
                  f"policy_loss={p_loss:.3f} value_loss={v_loss:.3f}")

        # CE eval
        if update % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                sample = random.sample(event_data, min(256,len(event_data)))
                texts, labels = zip(*sample)
                enc = tokenizer(list(texts), return_tensors="pt",
                                padding="max_length", truncation=True,
                                max_length=MAX_SEQ_LEN).to(DEVICE)
                logits, _ = model(enc.input_ids, enc.attention_mask)
                preds     = logits.argmax(dim=-1)
                acc       = (preds==torch.tensor(labels,device=DEVICE)).float().mean().item()
            model.train()
            logs["ce_evals"].append(update)
            logs["ce_accs"].append(acc)
            print(f"[Eval   {update:4d}] CE‑acc={acc:.3f}")

    print("\n✅ Training complete.\n")

    # 6) Plot metrics
    plt.figure()
    plt.plot(logs["updates"], logs["avg_rewards"])
    plt.title("Average Reward per PPO Update")
    plt.xlabel("Update"); plt.ylabel("Average Reward")
    plt.show()

    plt.figure()
    plt.plot(logs["updates"], logs["policy_losses"])
    plt.title("Policy Loss per PPO Update")
    plt.xlabel("Update"); plt.ylabel("Policy Loss")
    plt.show()

    plt.figure()
    plt.plot(logs["updates"], logs["value_losses"])
    plt.title("Value Loss per PPO Update")
    plt.xlabel("Update"); plt.ylabel("Value Loss")
    plt.show()

    plt.figure()
    plt.plot(logs["ce_evals"], logs["ce_accs"])
    plt.title("Cross‑Entropy Evaluation Accuracy")
    plt.xlabel("Update"); plt.ylabel("CE Accuracy")
    plt.show()

if __name__ == "__main__":
    main()
