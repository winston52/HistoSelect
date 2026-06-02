import argparse
import json
import os

import torch
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='Input VQA JSON (SlideChat schema)')
    ap.add_argument('--out',  required=True, help='Output .pt path')
    ap.add_argument('--ckpt', default='./data/models/conch_v1.pt')
    ap.add_argument('--gpu',  type=int, default=0)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Loading CONCH from {args.ckpt} → {device}")
    model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=args.ckpt, device=device)
    model.eval()
    tokenizer = get_tokenizer()

    with open(args.json) as f:
        data = json.load(f)

    ids, questions = [], []
    for item in data:
        if item.get('conversations') and item['conversations'][0]['from'] == 'human':
            ids.append(item['id'])
            questions.append(item['conversations'][0]['value'].replace('<image>\n', ''))
        else:
            print(f"  skipping malformed sample: id={item.get('id')}")

    print(f"Generating embeddings for {len(questions)} questions...")
    all_emb = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(questions), args.batch)):
            batch = questions[i:i + args.batch]
            toks = tokenize(texts=batch, tokenizer=tokenizer).to(device)
            emb = model.encode_text(toks)
            all_emb.append(emb.cpu())
    final = torch.cat(all_emb, dim=0)
    print(f"Final tensor shape: {final.shape}")

    embedding_map = {idx: emb for idx, emb in zip(ids, final)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(embedding_map, args.out)
    print(f"Saved {len(embedding_map)} embeddings → {args.out}")


if __name__ == '__main__':
    main()
