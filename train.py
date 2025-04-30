import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import math
import os
import time
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
import sacrebleu
from transformers import AutoConfig, AutoTokenizer, AutoModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LengthPredictor(nn.Module):
    def __init__(self, d_model: int, num_length_bins: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_length_bins)
        self.num_length_bins = num_length_bins

    def forward(self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_padding_mask.unsqueeze(-1)
        masked_encoder_output = encoder_output.permute(1, 0, 2) * mask
        summed = masked_encoder_output.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / valid_counts
        length_logits = self.linear(mean_pooled)
        return length_logits

class NATransformer(nn.Module):
    def __init__(self,
                 tgt_vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int = 5000,
                 pretrained_encoder_name: str = "facebook/mbart-large-50-many-to-many-mmt",
                 num_length_bins: int = 21):
        super().__init__()
        self.d_model = d_model

        print(f"Loading pretrained encoder: {pretrained_encoder_name}")
        try:
            pretrained_model = AutoModel.from_pretrained(pretrained_encoder_name)
            self.encoder = pretrained_model.encoder
            # config = AutoConfig.from_pretrained(pretrained_encoder_name)
            # assert d_model == config.hidden_size, f"d_model ({d_model}) must match pretrained encoder hidden size ({config.hidden_size})"
        except Exception as e:
            print(f"Failed to load pretrained model {pretrained_encoder_name}: {e}")
            raise
        
        print("Pretrained encoder loaded.")

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False, activation='relu')
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.length_predictor = LengthPredictor(d_model, num_length_bins)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src_input_ids: torch.Tensor,
                src_attention_mask: torch.Tensor,
                tgt_input_ids: torch.Tensor,
                tgt_padding_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            return_dict=True
        )
        memory = encoder_outputs.last_hidden_state
        memory = memory.permute(1, 0, 2)
        memory_key_padding_mask = (src_attention_mask == 0)

        length_logits = self.length_predictor(memory, memory_key_padding_mask)

        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model))

        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.output_projection(output)

        return logits, length_logits

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, src_lang_key, tgt_lang_key, max_seq_len=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.pad_token_id = tokenizer.pad_token_id
        # Use target language code as BOS for mBART style
        try:
             self.bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
        except AttributeError:
             # Fallback if tokenizer doesn't have tgt_lang or lang_code_to_id (e.g., not mBART tokenizer)
             self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 0 # Use 0 as a guess if undefined
             print(f"Warning: Using fallback BOS token ID: {self.bos_token_id}")

        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]['translation']
        src_text = item[self.src_lang_key]
        tgt_text = item[self.tgt_lang_key]

        # Tokenize source using HF tokenizer
        src_encoded = self.tokenizer(src_text, truncation=True, max_length=self.max_seq_len, padding=False, return_tensors=None)
        src_input_ids = src_encoded['input_ids']
        src_attention_mask = src_encoded['attention_mask'] # 1=real, 0=pad (will be padded in collate)

        # Tokenize target using HF tokenizer (for labels)
        # For mBART, target includes language code ID at the end
        tgt_encoded = self.tokenizer(text_target=tgt_text, truncation=True, max_length=self.max_seq_len, padding=False, return_tensors=None)
        tgt_ids = tgt_encoded['input_ids'] # Ground truth target

        src_len = len(src_input_ids)
        tgt_len = len(tgt_ids) # Length including BOS/EOS if added by tokenizer

        # Create placeholder input for NAT decoder: [BOS] + [PAD] * (tgt_len - 1)
        # Ensure tgt_len reflects the length we want the decoder to process
        tgt_input_ids = [self.bos_token_id] + [self.pad_token_id] * (tgt_len - 1) if tgt_len > 0 else []

        return {
            "src_input_ids": torch.tensor(src_input_ids, dtype=torch.long),
            "src_attention_mask": torch.tensor(src_attention_mask, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids[:self.max_seq_len], dtype=torch.long), # Truncate placeholder too
            "tgt_len": torch.tensor(tgt_len, dtype=torch.long),
            "src_len": torch.tensor(src_len, dtype=torch.long)
        }

def collate_fn_hf(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    # Find max lengths
    max_src_len = max(item["src_input_ids"].shape[0] for item in batch)
    max_tgt_len = max(item["tgt_ids"].shape[0] for item in batch)
    max_tgt_input_len = max(item["tgt_input_ids"].shape[0] for item in batch)
    # Ensure max_tgt_len >= max_tgt_input_len

    # Pad sequences
    src_ids_padded = []
    src_masks_padded = [] # HF mask (0=pad)
    tgt_ids_padded = []
    tgt_input_ids_padded = []
    tgt_masks_padded = [] # PyTorch mask (True=pad) for decoder input
    src_lens = []
    tgt_lens = []


    for item in batch:
        # Source padding (right-padded)
        src_len = item["src_input_ids"].shape[0]
        src_pad_len = max_src_len - src_len
        src_ids_padded.append(torch.cat([item["src_input_ids"], torch.full((src_pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        # HF attention mask (1=real, 0=pad)
        src_masks_padded.append(torch.cat([item["src_attention_mask"], torch.zeros(src_pad_len, dtype=torch.long)], dim=0))
        src_lens.append(item["src_len"])

        # Target padding (right-padded)
        tgt_len = item["tgt_ids"].shape[0]
        tgt_pad_len = max_tgt_len - tgt_len
        tgt_ids_padded.append(torch.cat([item["tgt_ids"], torch.full((tgt_pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        tgt_lens.append(item["tgt_len"])

        # Target Input padding (right-padded)
        tgt_input_len = item["tgt_input_ids"].shape[0]
        tgt_input_pad_len = max_tgt_input_len - tgt_input_len
        tgt_input_ids_padded.append(torch.cat([item["tgt_input_ids"], torch.full((tgt_input_pad_len,), pad_token_id, dtype=torch.long)], dim=0))

        # PyTorch target padding mask (True=pad) for decoder self-attention
        # Based on the length of the *decoder input*
        tgt_masks_padded.append(torch.cat([torch.zeros(tgt_input_len, dtype=torch.bool), torch.ones(tgt_input_pad_len, dtype=torch.bool)], dim=0))


    # Stack tensors
    # HF Encoder expects [Batch, Seq]
    src_ids_batch = torch.stack(src_ids_padded, dim=0)
    src_attention_mask_batch = torch.stack(src_masks_padded, dim=0)

    # Custom Decoder expects [Seq, Batch] by default
    # Ground Truth Targets [Seq, Batch] for loss calculation reshape later
    tgt_ids_batch = torch.stack(tgt_ids_padded, dim=1)
    # Decoder Input [Seq, Batch]
    tgt_input_ids_batch = torch.stack(tgt_input_ids_padded, dim=1)

    # PyTorch Masks expect [Batch, Seq]
    tgt_padding_mask_batch = torch.stack(tgt_masks_padded, dim=0)

    src_len_batch = torch.stack(src_lens, dim=0)
    tgt_len_batch = torch.stack(tgt_lens, dim=0)

    return {
        "src_input_ids": src_ids_batch,           # [batch, max_src_len]
        "src_attention_mask": src_attention_mask_batch, # [batch, max_src_len] (0=pad)
        "tgt_ids": tgt_ids_batch,                 # [max_tgt_len, batch] (Ground Truth)
        "tgt_input_ids": tgt_input_ids_batch,     # [max_tgt_input_len, batch] (Decoder Input)
        "tgt_padding_mask": tgt_padding_mask_batch, # [batch, max_tgt_input_len] (True=pad)
        "src_len": src_len_batch,                 # [batch]
        "tgt_len": tgt_len_batch                  # [batch]
    }

def train_epoch(model, dataloader, optimizer, criterion_token, criterion_length, device, pad_token_id,
                length_loss_weight=0.1, grad_clip=1.0,
                num_length_bins=21, length_bin_offset=10):
    model.train()
    total_loss = 0
    total_token_loss = 0
    total_length_loss = 0
    num_batches = len(dataloader)
    # progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, batch in enumerate(dataloader):
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        tgt_padding_mask = batch["tgt_padding_mask"].to(device)
        tgt_len = batch["tgt_len"].to(device)
        src_len = batch["src_len"].to(device)

        optimizer.zero_grad()

        logits, length_logits = model(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            tgt_input_ids=tgt_input_ids,
            tgt_padding_mask=tgt_padding_mask
        )

        loss_token = criterion_token(
            logits.view(-1, logits.shape[-1]),
            tgt_ids.view(-1)
        )

        length_diff = tgt_len - src_len
        target_length_bin = (length_diff + length_bin_offset).long() # Ensure long type
        target_length_bin = torch.clamp(target_length_bin, 0, num_length_bins - 1)

        loss_length = criterion_length(length_logits, target_length_bin)

        combined_loss = loss_token + length_loss_weight * loss_length

        combined_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += combined_loss.item()
        total_token_loss += loss_token.item()
        total_length_loss += loss_length.item()

        # progress_bar.set_postfix({
        #     "Batch Loss": f"{combined_loss.item():.4f}",
        #     "Avg Loss": f"{total_loss / (i+1):.4f}",
        #     "Token Loss": f"{loss_token.item():.4f}",
        #     "Length Loss": f"{loss_length.item():.4f}"
        #  })

    avg_loss = total_loss / num_batches
    avg_token_loss = total_token_loss / num_batches
    avg_length_loss = total_length_loss / num_batches
    return avg_loss, avg_token_loss, avg_length_loss


def evaluate(model, dataloader, tokenizer, device,
             num_length_bins=21, length_bin_offset=10):
    model.eval()
    hypotheses = []
    references_for_bleu = []

    pad_token_id = tokenizer.pad_token_id
    try:
        bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    except AttributeError:
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 0
    eos_token_id = tokenizer.eos_token_id

    print("\n--- Running Evaluation ---")

    with torch.no_grad():
        # progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(dataloader):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_ids_gt = batch["tgt_ids"].to(device)
            tgt_len_gt = batch["tgt_len"].to(device)
            src_len = batch["src_len"].to(device)

            try:
                # Replicate relevant parts of model forward for inference
                # 1. Encode source
                encoder_outputs = model.encoder(
                    input_ids=src_input_ids,
                    attention_mask=src_attention_mask,
                    return_dict=True
                )
                memory = encoder_outputs.last_hidden_state.permute(1, 0, 2)
                memory_key_padding_mask = (src_attention_mask == 0)

                # 2. Predict length
                length_logits = model.length_predictor(memory, memory_key_padding_mask)
                predicted_bin_index = length_logits.argmax(dim=-1)
                predicted_diff = predicted_bin_index - length_bin_offset
                predicted_lengths = src_len + predicted_diff
                max_src_len_batch = src_input_ids.size(1) # Use actual max source length from batch
                predicted_lengths = torch.clamp(predicted_lengths, min=2, max=max_src_len_batch + 50) # Clamp based on batch max_src_len

                # 3. Prepare decoder input
                max_pred_len = predicted_lengths.max().item()
                batch_size = src_input_ids.size(0)
                decoder_input_ids = torch.full((max_pred_len, batch_size), pad_token_id, dtype=torch.long, device=device)
                if max_pred_len > 0: decoder_input_ids[0, :] = bos_token_id
                tgt_padding_mask_gen = torch.arange(max_pred_len, device=device).unsqueeze(0).expand(batch_size, -1) >= predicted_lengths.unsqueeze(1)


                # 4. Embed decoder input
                tgt_emb = model.pos_encoder(model.tgt_embedding(decoder_input_ids) * math.sqrt(model.d_model))

                # 5. Decode
                output = model.decoder(
                    tgt=tgt_emb, memory=memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=tgt_padding_mask_gen,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

                # 6. Get predictions
                logits = model.output_projection(output)
                preds = logits.argmax(dim=-1)

            except Exception as e:
                print(f"\nERROR during generation in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # --- Detokenization ---
            try:
                preds_np = preds.cpu().numpy().T # [batch, seq]
                tgt_ids_gt_np = tgt_ids_gt.cpu().numpy().T if tgt_ids_gt.dim() > 1 else tgt_ids_gt.cpu().numpy() # Handle potential dimension issues if collate changes

                # Need to correctly handle tgt_ids_gt if its shape isn't [Seq, Batch] coming from collate
                # Assuming collate makes it [max_tgt_len, batch], transpose works.
                if tgt_ids_gt_np.ndim == 1 and batch_size == 1 : # Handle batch size 1 case
                    tgt_ids_gt_np = tgt_ids_gt_np.reshape(1,-1)


                for i in range(batch_size):
                    actual_pred_len = predicted_lengths[i].item()
                    pred_token_ids_for_item = preds_np[i, :actual_pred_len]

                    hypothesis = tokenizer.decode(pred_token_ids_for_item, skip_special_tokens=True)
                    hypotheses.append(hypothesis)

                    actual_gt_len = int(tgt_len_gt[i].item())
                    gt_token_ids_for_item = tgt_ids_gt_np[i, :actual_gt_len]
                    reference = tokenizer.decode(gt_token_ids_for_item, skip_special_tokens=True)
                    references_for_bleu.append([reference])

            except Exception as e:
                print(f"\nERROR during detokenization in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    bleu_score = 0.0
    if hypotheses and references_for_bleu and len(hypotheses) == len(references_for_bleu):
        print(f"Calculating BLEU score for {len(hypotheses)} pairs...")
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, references_for_bleu, lowercase=True)
            bleu_score = bleu.score
            print(f"\n--- BLEU Calculation Result ---\n{bleu}\n-----------------------------")
        except Exception as e:
             print(f"\nERROR during SacreBLEU calculation: {e}")
    elif not hypotheses:
         print("\nWARNING: No hypotheses were generated for BLEU calculation.")
    else:
         print(f"\nWARNING: Mismatch/Empty lists Hyp:{len(hypotheses)} Ref:{len(references_for_bleu)}. Cannot calculate BLEU.")
         
    print("------ Sample Sentences ------")
    print("Hypothesis: ", hypotheses[:5])
    print("\nReference: ", references_for_bleu[:5])
    print("-----------------------------")

    model.train()
    return bleu_score

def main():
    DATASET_NAME = "wmt14"
    DATASET_CONFIG = "fr-en"
    SRC_LANG = "fr"
    TGT_LANG = "en"
    PRETRAINED_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    SRC_LANG_CODE = "fr_XX"
    TGT_LANG_CODE = "en_XX"

    # Model Hyperparameters (Decoder / Length Predictor)
    N_HEADS = 8
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    MAX_POS_ENCODING = 1024 # Should match model max length capability

    # Length Prediction Configuration
    NUM_LENGTH_BINS = 21
    LENGTH_BIN_OFFSET = 10

    # Training Hyperparameters
    BATCH_SIZE = 8 # Adjust based on GPU memory (mBART-large is demanding)
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5 # Smaller LR typical for fine-tuning
    GRAD_CLIP = 1.0
    LENGTH_LOSS_WEIGHT = 0.1
    VALIDATION_SPLIT = 0.01 # e.g. 1%
    MAX_SEQ_LEN = 128 # Max sequence length for tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer and config for {PRETRAINED_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, src_lang=SRC_LANG_CODE, tgt_lang=TGT_LANG_CODE)
        config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    except Exception as e:
        print(f"Error loading pretrained model/tokenizer: {e}")
        return

    D_MODEL = config.hidden_size
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    print(f"Using Pretrained Model: {PRETRAINED_MODEL_NAME}")
    print(f"  d_model: {D_MODEL}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  pad_token_id: {pad_token_id}")

    print("Loading WMT14 fr-en dataset...")
    try:
        # Load only necessary splits, potentially smaller subset for demo
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True) # trust_remote_code might be needed
        # print(dataset)
        dataset['train']=dataset['train'].select(range(45000))

        if 'validation' not in dataset or len(dataset['validation']) == 0:
             print("Creating validation split from training data...")
             if len(dataset['train']) > 10000: # Only split if train set is reasonably large
                 train_test_split = dataset['train'].train_test_split(test_size=VALIDATION_SPLIT, seed=42)
                 dataset['train'] = train_test_split['train']
                 dataset['validation'] = train_test_split['test']
             else:
                 print("Training set too small to create validation split.")
                 dataset['validation'] = dataset['train'] # Use train as validation for quick test

        # Select smaller subset for faster testing if needed
        # dataset['train'] = dataset['train'].select(range(5000))
        # dataset['validation'] = dataset['validation'].select(range(500))

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")

    print("Creating datasets and dataloaders (using pretrained tokenizer)...")
    train_dataset = TranslationDataset(dataset['train'], tokenizer, SRC_LANG, TGT_LANG, max_seq_len=MAX_SEQ_LEN)
    val_dataset = TranslationDataset(dataset['validation'], tokenizer, SRC_LANG, TGT_LANG, max_seq_len=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn_hf(b, pad_token_id), num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=lambda b: collate_fn_hf(b, pad_token_id), num_workers=2, pin_memory=True)

    print("Initializing model...")
    # Note: src_vocab_size isn't strictly needed if src_embedding isn't used
    model = NATransformer(
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_POS_ENCODING,
        pretrained_encoder_name=PRETRAINED_MODEL_NAME,
        num_length_bins=NUM_LENGTH_BINS
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW often preferred for Transformers
    criterion_token = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_length = nn.CrossEntropyLoss()

    print("Starting training...")
    best_bleu = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        avg_loss, avg_token_loss, avg_length_loss = train_epoch(
            model, train_dataloader, optimizer, criterion_token, criterion_length, device, pad_token_id,
            LENGTH_LOSS_WEIGHT, GRAD_CLIP, NUM_LENGTH_BINS, LENGTH_BIN_OFFSET
        )
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {avg_loss:.4f} | Train Token Loss: {avg_token_loss:.4f} | Train Length Loss: {avg_length_loss:.4f}")

        epoch_bleu = 0.0
        if val_dataloader:
            epoch_bleu = evaluate(model, val_dataloader, tokenizer, device, NUM_LENGTH_BINS, LENGTH_BIN_OFFSET)
            print(f"\tValidation BLEU: {epoch_bleu:.2f}")

            if epoch_bleu > best_bleu:
                best_bleu = epoch_bleu
                # torch.save(model.state_dict(), 'nat_mbart_encoder_best_bleu.pt')
                # print(f"\tNew best BLEU score: {best_bleu:.2f}. Model saved.")
        else:
             print("\tSkipping validation BLEU calculation (no validation dataloader).")

        # torch.save(model.state_dict(), f'nat_mbart_encoder_epoch_{epoch}.pt')

    print("Training finished.")

    # save final model
    torch.save(model.state_dict(), 'NAT_mBART.pt')
    print("Model saved to directory.")

if __name__ == "__main__":
    main()