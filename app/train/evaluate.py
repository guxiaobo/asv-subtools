#!/usr/bin/env python3
"""
评估增量训练效果：对比 fine-tune 前后 embedding 的说话人区分度。

用法:
  PYTHONPATH=. /opt/anaconda3/bin/python train/evaluate.py --epoch 20
"""
import sys, os, json, argparse, logging, itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.fine_tune import (
    build_training_data, FBankExtractor,
    CAMPlus, ECAPA_TDNNSpeaker, ResNet34_2D,
    logger
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = _SCRIPT_DIR.parent / 'pytorch_weights'
MODELS = {
    'campplus': (CAMPlus, WEIGHTS / 'campplus_cn_common.pt', 192),
    'ecapa':    (ECAPA_TDNNSpeaker, WEIGHTS / 'avg_model.pt', 192),
    'resnet':   (ResNet34_2D, WEIGHTS / 'avg_model', 256),
}


def load_backbone(model_cls, ckpt_path, feat_dim, embedding_dim, fine_tuned_path=None):
    """Load model backbone (projection excluded)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cls(feat_dim=feat_dim, embedding_dim=embedding_dim)
    
    if fine_tuned_path and fine_tuned_path.exists():
        ckpt = torch.load(str(fine_tuned_path), map_location='cpu', weights_only=True)
        logger.info(f"  加载 fine-tuned: {fine_tuned_path.name}")
    else:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
        logger.info(f"  加载 pretrained: {ckpt_path.name}")
    
    # Filter out projection keys
    filtered = {k: v for k, v in ckpt.items()
                if not any(k.startswith(s) for s in ('projection.', 'classifier.'))}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def extract_embeddings(segments, fbank, model, device, max_frames=400):
    """Extract L2-normalized embeddings for all segments."""
    embeddings = []
    valid_indices = []
    
    for i, (wav_path, sid) in enumerate(segments):
        try:
            feat = fbank.extract_from_file(wav_path)
            if feat.size(0) < 20:  # too short
                continue
            # Crop/pad to max_frames
            if feat.size(0) > max_frames:
                # Center crop
                start = (feat.size(0) - max_frames) // 2
                feat = feat[start:start + max_frames]
            elif feat.size(0) < max_frames:
                pad = max_frames - feat.size(0)
                feat = torch.nn.functional.pad(feat, (0, 0, 0, pad))
            
            feat = feat.unsqueeze(0).to(device)  # (1, T, F)
            emb = model(feat, return_embedding=True)
            # L2 normalize
            emb = emb / emb.norm(dim=1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"  Skip {wav_path}: {e}")
    
    return np.array(embeddings), valid_indices


def compute_similarity_stats(embeddings, labels):
    """Compute within-class and between-class cosine similarity stats."""
    n = len(labels)
    if n < 2:
        return {'within': [], 'between': [], 'n_pairs_within': 0, 'n_pairs_between': 0}
    
    within_sims = []
    between_sims = []
    
    # Group by label
    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    
    for i, j in itertools.combinations(range(n), 2):
        sim = float(np.dot(embeddings[i], embeddings[j]))
        if labels[i] == labels[j]:
            within_sims.append(sim)
        else:
            between_sims.append(sim)
    
    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'n':    len(arr),
        } if arr else {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
    
    return {
        'within': stats(within_sims),
        'between': stats(between_sims),
    }


def evaluate_model(name, segments, id_to_speaker, fbank, epoch=20):
    """Evaluate one model before and after fine-tuning."""
    model_cls, ckpt_path, emb_dim = MODELS[name]
    fine_tuned_path = WEIGHTS / 'fine_tuned' / f'{name}_backbone.pt'
    
    logger.info(f'\n{"="*60}')
    logger.info(f'Model: {name}')
    logger.info(f'{"="*60}')
    
    labels = [sid for _, sid in segments]
    unique_speakers = len(set(labels))
    logger.info(f'  Segments: {len(segments)}, Speakers: {unique_speakers}')
    
    # ── Original pretrained model ──
    logger.info(f'  [Pretrained] Loading...')
    model_orig, device = load_backbone(model_cls, ckpt_path, 80, emb_dim)
    embs_orig, valid_idx = extract_embeddings(segments, fbank, model_orig, device)
    labels_orig = [labels[i] for i in valid_idx]
    stats_orig = compute_similarity_stats(embs_orig, labels_orig)
    
    within_m = stats_orig['within']['mean']
    between_m = stats_orig['between']['mean']
    sep = within_m - between_m
    logger.info(f'  [Pretrained]  Within={within_m:.4f}  Between={between_m:.4f}  Separation={sep:.4f}')
    
    # ── Fine-tuned model ──
    if fine_tuned_path.exists():
        logger.info(f'  [Fine-tuned] Loading {fine_tuned_path.name}...')
        model_ft, _ = load_backbone(model_cls, ckpt_path, 80, emb_dim, fine_tuned_path)
        embs_ft, valid_idx_ft = extract_embeddings(segments, fbank, model_ft, device)
        labels_ft = [labels[i] for i in valid_idx_ft]
        stats_ft = compute_similarity_stats(embs_ft, labels_ft)
        
        within_m_ft = stats_ft['within']['mean']
        between_m_ft = stats_ft['between']['mean']
        sep_ft = within_m_ft - between_m_ft
        logger.info(f'  [Fine-tuned] Within={within_m_ft:.4f}  Between={between_m_ft:.4f}  Separation={sep_ft:.4f}')
        logger.info(f'  [Delta]      Within Δ={within_m_ft-within_m:+.4f}  Between Δ={between_m_ft-between_m:+.4f}  Sep Δ={sep_ft-sep:+.4f}')
        
        # Per-speaker analysis
        logger.info(f'\n  Per-speaker within-class similarity (Fine-tuned vs Pretrained):')
        groups_ft = defaultdict(list)
        for i, lbl in enumerate(labels_ft):
            groups_ft[lbl].append(i)
        
        total_improve = 0
        total_degrade = 0
        for lbl_id, indices in sorted(groups_ft.items(), key=lambda x: -len(x[1])):
            speaker_name = id_to_speaker[lbl_id]
            n = len(indices)
            if n < 2:
                continue
            
            # Within-class mean similarity for this speaker
            sims_orig = []
            sims_ft = []
            for i, j in itertools.combinations(indices, 2):
                sims_orig.append(float(np.dot(embs_orig[i], embs_orig[j])))
                sims_ft.append(float(np.dot(embs_ft[i], embs_ft[j])))
            
            m_orig = np.mean(sims_orig) if sims_orig else 0
            m_ft = np.mean(sims_ft) if sims_ft else 0
            delta = m_ft - m_orig
            arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
            if delta > 0:
                total_improve += 1
            elif delta < 0:
                total_degrade += 1
            logger.info(f'    {speaker_name:20s} (n={n:2d}): orig={m_orig:.4f}  ft={m_ft:.4f}  Δ={delta:+.4f} {arrow}')
        
        logger.info(f'\n  Summary: improved={total_improve} degraded={total_degrade}')
        
        return {
            'name': name,
            'pretrained': stats_orig,
            'fine_tuned': stats_ft,
            'separation_delta': sep_ft - sep,
            'improved': total_improve,
            'degraded': total_degrade,
        }
    else:
        logger.warning(f'  No fine-tuned model found at {fine_tuned_path}')
        return {
            'name': name,
            'pretrained': stats_orig,
            'fine_tuned': None,
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuning effect')
    parser.add_argument('--model', default='all', choices=['all', 'campplus', 'ecapa', 'resnet'])
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    
    # Build the same training data
    logger.info('Building training data...')
    segments, id_to_speaker = build_training_data()
    logger.info(f'Total: {len(segments)} segments, {len(id_to_speaker)} speakers')
    
    fbank = FBankExtractor()
    
    models_to_test = ['campplus', 'ecapa', 'resnet'] if args.model == 'all' else [args.model]
    
    results = {}
    for name in models_to_test:
        r = evaluate_model(name, segments, id_to_speaker, fbank, epoch=args.epoch)
        results[name] = r
    
    logger.info(f'\n{"="*60}')
    logger.info('FINAL SUMMARY')
    logger.info(f'{"="*60}')
    logger.info(f'{"Model":12s} {"Within(orig)":12s} {"Within(ft)":12s} {"Between(orig)":12s} {"Between(ft)":12s} {"Sep Δ":10s} {"Imp":4s} {"Deg":4s}')
    logger.info('-' * 70)
    for name in models_to_test:
        r = results[name]
        p = r['pretrained']
        f = r.get('fine_tuned')
        if f:
            logger.info(f'{name:12s} {p["within"]["mean"]:.4f}        {f["within"]["mean"]:.4f}        '
                        f'{p["between"]["mean"]:.4f}        {f["between"]["mean"]:.4f}        '
                        f'{r["separation_delta"]:+.4f}     {r["improved"]:3d}  {r["degraded"]:3d}')
        else:
            logger.info(f'{name:12s} {p["within"]["mean"]:.4f}        N/A             '
                        f'{p["between"]["mean"]:.4f}        N/A             N/A       N/A  N/A')

    # ── Persist evaluation results to SQLite ──
    _save_eval_results(results)
    logger.info("Evaluation results saved to model_versions table.")


def _save_eval_results(results: Dict) -> None:
    """Write evaluation results from evaluate.py to model_versions table.

    Each model's stats (within/between similarity, per-speaker breakdown)
    are stored in the ``metrics`` JSON column.
    """
    import json
    from datetime import datetime as dt

    from train.db import get_connection, insert_model_version, init_db

    conn = get_connection()
    init_db(conn)

    for name, r in results.items():
        if r.get('fine_tuned') is None:
            logger.info("  Skip %s: no fine-tuned model found", name)
            continue

        name_map = {'campplus': 'CAM++', 'ecapa': 'ECAPA', 'resnet': 'ResNet34'}
        display_name = name_map.get(name, name)
        emb_dim = 192 if name != 'resnet' else 256

        sep_delta = r.get('separation_delta', 0.0)
        improved = r.get('improved', 0)
        degraded = r.get('degraded', 0)

        metrics = {
            'eval_type': 'similarity_stats',
            'pretrained': r['pretrained'],
            'fine_tuned': r['fine_tuned'],
            'separation_delta': sep_delta,
            'improved': improved,
            'degraded': degraded,
        }

        tag = dt.now().strftime(f"{name}_eval_%Y%m%d_%H%M%S")

        insert_model_version(
            conn,
            model_name=display_name,
            version_tag=tag,
            version=tag,
            embedding_dim=emb_dim,
            eval_metric='separation_delta',
            eval_value=sep_delta,
            improved=True,
            model_path='',
            base_model=name,
            config='{}',
            metrics=json.dumps(metrics, ensure_ascii=False),
            notes=f"Evaluate: sep_delta={sep_delta:+.4f}, improved={improved}, degraded={degraded}",
            score=sep_delta,
        )

    conn.close()


if __name__ == '__main__':
    main()
