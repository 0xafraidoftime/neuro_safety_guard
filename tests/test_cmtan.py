"""
test_cmtan.py — Unit tests for the CMTAN model and end-to-end pipeline.
Run with: pytest tests/test_cmtan.py -v
"""

import pytest
import torch
import numpy as np

from src.models.fnirs_encoder import FNIRSEncoder
from src.models.eeg_encoder import EEGEncoder, BandPowerExtractor
from src.models.cmtan import CMTAN, CrossModalAttentionBlock
from src.models.classifier import NeuroSafetyGuard, make_synthetic_batch
from src.utils.personalizer import AdaptiveInterventionController, CognitiveState


# ── Configuration ─────────────────────────────────────────────────────────────
BATCH = 4
T = 5          # 5-second window
N_FNIRS = 20
N_EEG = 32
SFREQ = 256
EMBED = 64     # smaller for tests


# ── fNIRS Encoder ─────────────────────────────────────────────────────────────
class TestFNIRSEncoder:
    def test_output_shape(self):
        enc = FNIRSEncoder(n_fnirs_ch=N_FNIRS, embed_dim=EMBED)
        x = torch.randn(BATCH, N_FNIRS * 2, T)
        out = enc(x)
        assert out.shape == (BATCH, T, EMBED), f"Expected {(BATCH, T, EMBED)}, got {out.shape}"

    def test_no_nan(self):
        enc = FNIRSEncoder(n_fnirs_ch=N_FNIRS, embed_dim=EMBED)
        x = torch.randn(BATCH, N_FNIRS * 2, T)
        out = enc(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        enc = FNIRSEncoder(n_fnirs_ch=N_FNIRS, embed_dim=EMBED)
        x = torch.randn(BATCH, N_FNIRS * 2, T, requires_grad=False)
        out = enc(x)
        loss = out.mean()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ── Band Power Extractor ───────────────────────────────────────────────────────
class TestBandPower:
    def test_output_shape(self):
        bp = BandPowerExtractor(n_eeg_ch=N_EEG, sfreq=SFREQ, epoch_len=SFREQ)
        x = torch.randn(BATCH, N_EEG, SFREQ)
        out = bp(x)
        assert out.shape == (BATCH, N_EEG * 5)  # 5 bands

    def test_positive_power(self):
        bp = BandPowerExtractor(n_eeg_ch=N_EEG, sfreq=SFREQ, epoch_len=SFREQ)
        x = torch.randn(BATCH, N_EEG, SFREQ)
        out = bp(x)
        assert (out >= 0).all(), "Log band power should be non-negative"


# ── EEG Encoder ───────────────────────────────────────────────────────────────
class TestEEGEncoder:
    def test_output_shape(self):
        enc = EEGEncoder(n_eeg_ch=N_EEG, sfreq=SFREQ, epoch_len=SFREQ, embed_dim=EMBED)
        x = torch.randn(BATCH, T, N_EEG, SFREQ)
        out = enc(x)
        assert out.shape == (BATCH, T, EMBED)


# ── CMTAN ─────────────────────────────────────────────────────────────────────
class TestCMTAN:
    def test_output_shape(self):
        model = CMTAN(embed_dim=EMBED, n_layers=2, n_heads=4, n_classes=4)
        fnirs = torch.randn(BATCH, T, EMBED)
        eeg   = torch.randn(BATCH, T, EMBED)
        logits, attn_maps = model(fnirs, eeg)
        assert logits.shape == (BATCH, 4)
        assert len(attn_maps) == 2

    def test_attention_weights_sum_to_one(self):
        model = CMTAN(embed_dim=EMBED, n_layers=1, n_heads=4, n_classes=4)
        fnirs = torch.randn(BATCH, T, EMBED)
        eeg   = torch.randn(BATCH, T, EMBED)
        _, attn_maps = model(fnirs, eeg)
        # attn_maps[0]: (batch, T, T)
        row_sums = attn_maps[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_pool_modes(self):
        for pool in ("mean", "last", "cls"):
            model = CMTAN(embed_dim=EMBED, n_layers=1, n_heads=4, n_classes=4, pool=pool)
            fnirs = torch.randn(BATCH, T, EMBED)
            eeg   = torch.randn(BATCH, T, EMBED)
            logits, _ = model(fnirs, eeg)
            assert logits.shape == (BATCH, 4), f"Failed for pool={pool}"


# ── End-to-End ────────────────────────────────────────────────────────────────
class TestEndToEnd:
    def test_forward_pass(self):
        model = NeuroSafetyGuard(n_fnirs_ch=N_FNIRS, n_eeg_ch=N_EEG, embed_dim=EMBED)
        fnirs, eeg, labels = make_synthetic_batch(batch=BATCH, window_s=T,
                                                   n_fnirs_ch=N_FNIRS, n_eeg_ch=N_EEG)
        logits, _ = model(fnirs, eeg)
        assert logits.shape == (BATCH, 4)

    def test_predict(self):
        model = NeuroSafetyGuard(n_fnirs_ch=N_FNIRS, n_eeg_ch=N_EEG, embed_dim=EMBED)
        fnirs, eeg, _ = make_synthetic_batch(batch=2, window_s=T, n_fnirs_ch=N_FNIRS, n_eeg_ch=N_EEG)
        preds, confs, probs = model.predict(fnirs, eeg)
        assert preds.shape == (2,)
        assert ((confs >= 0) & (confs <= 1)).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


# ── Adaptive Controller ────────────────────────────────────────────────────────
class TestAdaptiveController:
    def test_no_intervention_on_normal(self):
        ctrl = AdaptiveInterventionController()
        ctrl.register_worker("W001")
        probs = np.array([0.9, 0.05, 0.03, 0.02])
        result = ctrl.process("W001", probs, pred_state=0, confidence=0.9)
        assert result["intervene"] == False

    def test_intervention_on_tunneling(self):
        ctrl = AdaptiveInterventionController(
            confidence_threshold=0.5, calibration_windows=1
        )
        ctrl.register_worker("W002")
        probs = np.array([0.05, 0.05, 0.85, 0.05])
        result = ctrl.process("W002", probs, pred_state=2, confidence=0.85)
        assert result["intervene"] == True
        assert "peripheral" in result["intervention"].lower()

    def test_cooldown(self):
        ctrl = AdaptiveInterventionController(
            confidence_threshold=0.5, calibration_windows=1
        )
        ctrl.register_worker("W003")
        ctrl.workers["W003"].cooldown_s = 9999  # effectively infinite
        probs = np.array([0.05, 0.05, 0.85, 0.05])
        r1 = ctrl.process("W003", probs, pred_state=2, confidence=0.85)
        r2 = ctrl.process("W003", probs, pred_state=2, confidence=0.85)
        assert r1["intervene"] == True
        assert r2["intervene"] == False  # blocked by cooldown
