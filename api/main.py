"""Fed-XRay FastAPI Async Backend Application.

High-performance REST & Server-Sent Events (SSE) streaming API for the Next.js frontend.
Supports:
- Vision Transformer & CNN backbones
- Parameter-Efficient Fine-Tuning (FFA-LoRA, FedSA-LoRA)
- Modern FL Optimizers: FedDyn, FedOpt/FedAdam, SCAFFOLD, MOON, FedProx
- Strategy E Real-World Multi-Center Cohorts & Scenarios A-G
- Evidence-Grounded Federated RAG & Grad-CAM XAI
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import time
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.fed_xray.data.real_world import (
    RealWorldPatientRecord,
    StrategyEDatasetEcosystem,
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.cdss.xai import GradCAM, GradCAMUnavailableError, get_explanation_text
from src.fed_xray.cdss.similarity import (
    FederatedRAGCaseMatcher,
    extract_embedding,
    LABEL_NAMES,
)
from src.fed_xray.cdss.voice import get_or_create_audio
from src.fed_xray.cdss.report import generate_medical_report

from api.schemas import (
    CohortGenerateRequest,
    CohortGenerateResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HospitalCohortInfo,
    RagSimilarResponse,
    RagTwinCase,
)

app = FastAPI(
    title="Fed-XRay Clinical AI API",
    description="Federated Vision Transformers & Multimodal CDSS API",
    version="2.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloaders: Optional[List[Any]] = None
        self.global_test_set: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.hospital_samples: Dict[int, Any] = {}
        self.server: Optional[CentralServer] = None
        self.model_trained: bool = False
        self.model_type: str = "vit_tiny"
        self.peft_mode: Optional[str] = "ffa_lora"
        self.rag_matcher: Optional[FederatedRAGCaseMatcher] = None
        self.last_scan: Optional[np.ndarray] = None
        self.reference_records: List[RealWorldPatientRecord] = []
        self.data_source: str = "none"
        self.training_history: Dict[str, Any] = {
            "loss": [],
            "accuracy": [],
            "round": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "test_accuracy": [],
            "test_loss": [],
            "blocked_count": 0,
        }
        # Background training task state (for Render/proxy SSE fallback)
        self.task_running: bool = False
        self.task_rounds: List[Dict] = []
        self.task_total_rounds: int = 5
        self.task_current_round: int = 0
        self.task_complete: bool = False
        self.task_error: Optional[str] = None


state = AppState()

STRATEGY_E_CENTERS = [
    "BCN-20000 Skin Cancer Hub (Barcelona)",
    "ViDIR Dermatopathology Institute (Vienna)",
    "Queensland Oncology Screening Center",
    "Beth Israel Deaconess Medical Center",
    "NCT National Center for Tumor Diseases",
    "UMM Mannheim Medical Center",
    "Stanford Thoracic Oncology Center",
    "Charite Comprehensive Cancer Center",
]

def require_cohort() -> Tuple[List[Any], Tuple[torch.Tensor, torch.Tensor]]:
    """Return the loaded cohort or fail explicitly; no cohort is ever fabricated implicitly."""
    if state.dataloaders is None or state.global_test_set is None:
        raise HTTPException(
            status_code=409,
            detail="No cohort is loaded. Generate or load a cohort (POST /api/cohorts/generate) before training.",
        )
    return state.dataloaders, state.global_test_set


def require_trained_server() -> CentralServer:
    """Return the trained federated server or fail explicitly if training has not completed."""
    if not state.model_trained or state.server is None:
        raise HTTPException(
            status_code=409,
            detail="Federated model is not trained yet. Run federated training (GET /api/fl/train-stream) before diagnosing.",
        )
    return state.server


def resolve_data_path(name: str) -> Path:
    """Resolve a client-supplied relative path inside the configured data directory."""
    base = os.environ.get("FED_XRAY_DATA_DIR")
    if not base:
        raise HTTPException(status_code=400, detail="FED_XRAY_DATA_DIR is not configured on the server.")
    base_path = Path(base).resolve()
    target = (base_path / name).resolve()
    if base_path != target and base_path not in target.parents:
        raise HTTPException(status_code=400, detail="Path escapes the configured data directory.")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found in data directory: {name}")
    return target


def build_rag_matcher(server: CentralServer) -> FederatedRAGCaseMatcher:
    """Build the case bank from the loaded training cohort and the trained encoder."""
    if not state.reference_records:
        raise HTTPException(status_code=409, detail="No reference cohort is loaded for case retrieval.")
    return FederatedRAGCaseMatcher(reference_records=state.reference_records, model=server.get_model())



@app.get("/")
@app.head("/")
async def root_index():
    return {
        "status": "healthy",
        "service": "Fed-XRay SOTA ViT-FL & Multimodal CDSS Engine",
        "version": "2.1.0",
        "device": str(state.device),
        "model_trained": state.model_trained,
        "active_architecture": state.model_type,
        "active_peft": state.peft_mode,
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": str(state.device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/state")
async def get_state():
    return {
        "model_trained": state.model_trained,
        "hospitals_loaded": len(state.hospital_samples),
        "has_test_set": state.global_test_set is not None,
        "history_rounds": len(state.training_history["round"]),
        "model_type": state.model_type,
        "peft_mode": state.peft_mode,
    }


@app.post("/api/cohorts/generate", response_model=CohortGenerateResponse)
async def generate_cohorts(req: CohortGenerateRequest):
    """Generate multi-center real-world cohort partitions adhering to Strategy E scenarios."""
    ecosystem = StrategyEDatasetEcosystem(seed=42)
    if req.manifest_name:
        manifest_path = resolve_data_path(req.manifest_name)
        images_dir = resolve_data_path(req.images_dir_name) if req.images_dir_name else None
        try:
            records = ecosystem.load_from_disk_manifest(
                manifest_path=str(manifest_path),
                images_dir=str(images_dir) if images_dir else None,
                target_size=(28, 28),
                in_channels=1,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Manifest could not be loaded: {exc}")
        if not records:
            raise HTTPException(status_code=422, detail="Manifest contains no readable records.")
        if any(r.label not in (0, 1, 2) for r in records):
            raise HTTPException(status_code=422, detail="Manifest labels must be integers 0, 1 or 2.")
        data_source = "disk_manifest"
    else:
        records = ecosystem.generate_synthetic_realworld_cohort(
            dataset_name=req.dataset_name,
            num_patients=max(60, req.num_hospitals * 25),
            samples_per_patient=max(1, req.samples_per_hospital // 25),
            num_classes=3,
        )
        data_source = "synthetic_phantom"

    # Leak-free partitioning
    train_recs, val_recs, test_recs = ecosystem.leak_free_patient_split(
        records=records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    client_partitions = ecosystem.partition_into_scenarios(
        records=train_recs,
        num_clients=req.num_hospitals,
        scenario=req.scenario,
        num_classes=3,
    )

    state.hospital_samples = {}
    hospitals_info: List[HospitalCohortInfo] = []
    dataloaders_list: List[Any] = []

    for h in range(req.num_hospitals):
        part = client_partitions[h]
        loader = ecosystem.records_to_dataloader(part, batch_size=32, shuffle=True)
        dataloaders_list.append(loader)

        images = np.stack([r.image_tensor.numpy()[0] for r in part])
        labels = np.array([r.label for r in part])

        c0 = int(np.sum(labels == 0))
        c1 = int(np.sum(labels == 1))
        c2 = int(np.sum(labels == 2))
        total_p = max(1, len(labels))

        dist = [c0 / total_p, c1 / total_p, c2 / total_p]
        counts = {"normal": c0, "pneumonia": c1, "covid": c2}

        sample_imgs = [r.image_tensor.numpy()[0].tolist() for r in part[:9]]
        sample_lbls = [r.label for r in part[:9]]

        if data_source == "disk_manifest" and part and any(r.site_id for r in part):
            inst_name = Counter(r.site_id for r in part).most_common(1)[0][0]
        else:
            inst_name = STRATEGY_E_CENTERS[h % len(STRATEGY_E_CENTERS)]

        info = HospitalCohortInfo(
            hospital_id=h + 1,
            name=inst_name,
            num_samples=len(part),
            distribution=[round(x, 4) for x in dist],
            counts=counts,
            sample_images=sample_imgs,
            sample_labels=sample_lbls,
        )
        hospitals_info.append(info)

        state.hospital_samples[h] = {
            "images": images,
            "labels": labels,
            "distribution": dist,
        }

    state.dataloaders = dataloaders_list
    state.reference_records = list(train_recs)
    state.data_source = data_source
    state.rag_matcher = None
    state.model_trained = False

    # Global hold-out test set from leak-free split
    test_images = torch.stack([r.image_tensor for r in test_recs])
    test_labels = torch.tensor([r.label for r in test_recs], dtype=torch.long)
    state.global_test_set = (test_images, test_labels)

    total_samples = sum(len(p) for p in client_partitions)

    return CohortGenerateResponse(
        success=True,
        num_hospitals=req.num_hospitals,
        total_samples=total_samples,
        data_source=data_source,
        hospitals=hospitals_info,
    )


@app.get("/api/fl/train-stream")
async def stream_federated_training(
    num_rounds: int = Query(default=5, ge=1, le=20),
    local_epochs: int = Query(default=2, ge=1, le=5),
    learning_rate: float = Query(default=0.0001),
    privacy_noise: float = Query(default=0.01),
    simulate_attack: bool = Query(default=False),
    activate_defense: bool = Query(default=True),
    algorithm: str = Query(default="FedDyn"),
    loss_fn: str = Query(default="DAFL"),
    model_type: str = Query(default="vit_tiny"),
    peft_mode: str = Query(default="ffa_lora"),
    secure_aggregation: bool = Query(default=False),
    use_ckks: bool = Query(default=False),
):
    """Server-Sent Events (SSE) streaming federated ViT training in real time."""
    active_dataloaders, active_global_test_set = require_cohort()

    actual_peft = None if peft_mode.lower() in ("none", "false", "off") else peft_mode
    state.model_type = model_type
    state.peft_mode = peft_mode

    async def event_generator():
        n_hospitals = len(active_dataloaders)
        state.server = CentralServer(
            device=state.device,
            privacy_noise=privacy_noise,
            defense_mode=activate_defense,
            model_type=model_type,
            peft_mode=actual_peft,
            lora_r=8,
            lora_alpha=16.0,
            deep_layers_only=True,
            secure_aggregation=secure_aggregation,
            use_ckks=use_ckks,
        )

        clients = []
        for i in range(n_hospitals):
            is_malicious = simulate_attack and i == 2
            clients.append(
                HospitalClient(
                    client_id=i,
                    dataloader=active_dataloaders[i],
                    device=state.device,
                    learning_rate=learning_rate,
                    local_epochs=local_epochs,
                    malicious=is_malicious,
                    model_type=model_type,
                    peft_mode=actual_peft,
                    lora_r=8,
                    lora_alpha=16.0,
                    deep_layers_only=True,
                )
            )

        test_images, test_labels = active_global_test_set

        state.training_history = {
            "loss": [],
            "accuracy": [],
            "round": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "test_accuracy": [],
            "test_loss": [],
            "blocked_count": 0,
        }

        for round_num in range(1, num_rounds + 1):
            metrics, client_metrics, test_metrics, sec_report = run_federated_round(
                server=state.server,
                clients=clients,
                round_num=round_num,
                total_rounds=num_rounds,
                test_images=test_images,
                test_labels=test_labels,
                use_defense=activate_defense,
                algorithm=algorithm,
                loss_fn_name=loss_fn,
                enable_prototypes=True,
                proto_weight=0.1,
                prox_mu=0.01 if algorithm.upper() in ("FEDPROX", "PROX") else 0.0,
                feddyn_alpha=0.01 if algorithm.upper() == "FEDDYN" else 0.0,
                secure_aggregation=secure_aggregation,
                use_ckks=use_ckks,
            )

            t_loss = float(metrics["loss"])
            t_acc = float(metrics["accuracy"] * 100.0)

            val_acc = float(test_metrics.accuracy * 100.0) if test_metrics else t_acc
            val_loss = float(test_metrics.loss) if test_metrics else t_loss
            prec = float(test_metrics.precision * 100.0) if test_metrics else 0.0
            rec = float(test_metrics.recall * 100.0) if test_metrics else 0.0
            f1 = float(test_metrics.f1_score * 100.0) if test_metrics else 0.0

            blocked_nodes = sec_report.clients_blocked if sec_report else []
            threat_detected = len(blocked_nodes) > 0

            state.training_history["round"].append(round_num)
            state.training_history["loss"].append(val_loss)
            state.training_history["accuracy"].append(val_acc)
            state.training_history["test_accuracy"].append(val_acc)
            state.training_history["test_loss"].append(val_loss)
            state.training_history["f1_score"].append(f1)

            data_payload = {
                "round_num": round_num,
                "total_rounds": num_rounds,
                "train_loss": round(t_loss, 4),
                "train_accuracy": round(t_acc, 2),
                "test_loss": round(val_loss, 4),
                "test_accuracy": round(val_acc, 2),
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1_score": round(f1, 2),
                "threat_detected": threat_detected,
                "blocked_nodes": [n + 1 for n in blocked_nodes],
                "status": "training" if round_num < num_rounds else "complete",
                "model_type": model_type,
                "peft_mode": peft_mode,
                "privacy_epsilon": round(state.server.get_privacy_spent(), 4) if state.server else 0.0,
                "uplink_bytes_per_client": int(metrics["uplink_bytes_per_client"]),
                "full_model_bytes": int(metrics["full_model_bytes"]),
            }

            if round_num >= num_rounds:
                state.model_trained = True
                if state.server is not None and state.reference_records:
                    state.rag_matcher = build_rag_matcher(state.server)

            yield f"data: {json.dumps(data_payload)}\n\n"
            await asyncio.sleep(0.05)

        state.model_trained = True
        if state.server is not None and state.reference_records and state.rag_matcher is None:
            state.rag_matcher = build_rag_matcher(state.server)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/api/fl/train-start")
async def start_training_background(
    num_rounds: int = Query(default=5, ge=1, le=20),
    local_epochs: int = Query(default=2, ge=1, le=5),
    learning_rate: float = Query(default=0.0001),
    simulate_attack: bool = Query(default=False),
    activate_defense: bool = Query(default=True),
    algorithm: str = Query(default="FedDyn"),
    loss_fn: str = Query(default="DAFL"),
    model_type: str = Query(default="vit_tiny"),
    peft_mode: str = Query(default="ffa_lora"),
    secure_aggregation: bool = Query(default=False),
    use_ckks: bool = Query(default=False),
):
    """Launch federated training as a background thread (Render SSE proxy fallback)."""
    require_cohort()
    if state.task_running:
        return {"status": "already_running", "current_round": state.task_current_round}

    # Reset task state
    state.task_running = True
    state.task_rounds = []
    state.task_total_rounds = num_rounds
    state.task_current_round = 0
    state.task_complete = False
    state.task_error = None

    def run_training():
        try:
            dataloaders, (test_images, test_labels) = require_cohort()

            actual_peft = None if peft_mode.lower() in ("none", "false", "off") else peft_mode
            state.model_type = model_type
            state.peft_mode = peft_mode

            srv = CentralServer(
                device=state.device,
                defense_mode=activate_defense,
                model_type=model_type,
                peft_mode=actual_peft,
                lora_r=8,
                lora_alpha=16.0,
                deep_layers_only=True,
                secure_aggregation=secure_aggregation,
                use_ckks=use_ckks,
            )
            state.server = srv

            clients = [
                HospitalClient(
                    client_id=i,
                    dataloader=dataloaders[i],
                    device=state.device,
                    learning_rate=learning_rate,
                    local_epochs=local_epochs,
                    malicious=simulate_attack and i == 2,
                    model_type=model_type,
                    peft_mode=actual_peft,
                    lora_r=8,
                    lora_alpha=16.0,
                    deep_layers_only=True,
                )
                for i in range(len(dataloaders))
            ]

            for round_num in range(1, num_rounds + 1):
                metrics, _, test_metrics, sec_report = run_federated_round(
                    server=srv,
                    clients=clients,
                    round_num=round_num,
                    total_rounds=num_rounds,
                    test_images=test_images,
                    test_labels=test_labels,
                    use_defense=activate_defense,
                    algorithm=algorithm,
                    loss_fn_name=loss_fn,
                    enable_prototypes=True,
                    proto_weight=0.1,
                    prox_mu=0.01 if algorithm.upper() in ("FEDPROX", "PROX") else 0.0,
                    feddyn_alpha=0.01 if algorithm.upper() == "FEDDYN" else 0.0,
                    secure_aggregation=secure_aggregation,
                    use_ckks=use_ckks,
                )

                t_loss = float(metrics["loss"])
                t_acc = float(metrics["accuracy"] * 100.0)
                val_acc = float(test_metrics.accuracy * 100.0) if test_metrics else t_acc
                val_loss = float(test_metrics.loss) if test_metrics else t_loss
                prec = float(test_metrics.precision * 100.0) if test_metrics else 0.0
                rec = float(test_metrics.recall * 100.0) if test_metrics else 0.0
                f1 = float(test_metrics.f1_score * 100.0) if test_metrics else 0.0
                blocked_nodes = sec_report.clients_blocked if sec_report else []

                round_data = {
                    "round_num": round_num,
                    "total_rounds": num_rounds,
                    "train_loss": round(t_loss, 4),
                    "train_accuracy": round(t_acc, 2),
                    "test_loss": round(val_loss, 4),
                    "test_accuracy": round(val_acc, 2),
                    "precision": round(prec, 2),
                    "recall": round(rec, 2),
                    "f1_score": round(f1, 2),
                    "threat_detected": len(blocked_nodes) > 0,
                    "blocked_nodes": [n + 1 for n in blocked_nodes],
                    "status": "training" if round_num < num_rounds else "complete",
                    "model_type": model_type,
                    "peft_mode": peft_mode,
                    "privacy_epsilon": round(srv.get_privacy_spent(), 4),
                    "uplink_bytes_per_client": int(metrics["uplink_bytes_per_client"]),
                    "full_model_bytes": int(metrics["full_model_bytes"]),
                }
                state.task_rounds.append(round_data)
                state.task_current_round = round_num

            state.model_trained = True
            state.rag_matcher = build_rag_matcher(srv)
            state.task_complete = True

        except Exception as e:
            state.task_error = str(e)
            state.task_complete = True
        finally:
            state.task_running = False

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return {"status": "started", "total_rounds": num_rounds}


@app.get("/api/fl/train-status")
async def get_training_status(from_round: int = Query(default=0)):
    """Poll training progress - returns new rounds since from_round (Render proxy fallback)."""
    new_rounds = state.task_rounds[from_round:]
    return {
        "is_running": state.task_running,
        "is_complete": state.task_complete,
        "current_round": state.task_current_round,
        "total_rounds": state.task_total_rounds,
        "error": state.task_error,
        "new_rounds": new_rounds,
    }


@app.post("/api/cdss/diagnose", response_model=DiagnoseResponse)
async def perform_diagnosis(req: DiagnoseRequest):
    """Diagnose a scan with the trained federated model and compute a Grad-CAM explanation."""
    server = require_trained_server()
    model = server.get_model()
    model.eval()

    true_class = -1
    if req.image_base64:
        try:
            header, _, encoded = req.image_base64.partition(",")
            raw_b64 = encoded if encoded else header
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            pil_img = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
            image_np = (np.array(pil_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}")
        image_source = "uploaded"
    elif req.image_data:
        try:
            raw_arr = np.array(req.image_data, dtype=np.float32)
            if raw_arr.ndim != 2:
                raise ValueError("image_data must be a 2D matrix")
            t_img = torch.tensor(raw_arr).unsqueeze(0).unsqueeze(0)
            if t_img.shape[-2:] != (28, 28):
                t_img = F.interpolate(t_img, size=(28, 28), mode="bilinear", align_corners=False)
            image_np = t_img.squeeze().cpu().numpy().clip(0.0, 1.0)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image_data matrix: {exc}")
        image_source = "uploaded"
    else:
        if state.global_test_set is None:
            raise HTTPException(status_code=409, detail="No held-out test set is loaded and no image was supplied.")
        test_images, test_labels = state.global_test_set
        if req.class_index is not None and req.class_index in (0, 1, 2):
            candidates = torch.nonzero(test_labels == req.class_index).flatten()
            if candidates.numel() == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"The held-out test set contains no sample of class {req.class_index}.",
                )
        else:
            candidates = torch.arange(test_labels.numel())
        pick = int(candidates[int(np.random.randint(0, candidates.numel()))])
        image_np = test_images[pick, 0].cpu().numpy()
        true_class = int(test_labels[pick])
        image_source = f"holdout_test_set:{state.data_source}"

    if req.class_index is not None and req.class_index in (0, 1, 2) and true_class == -1:
        true_class = req.class_index

    state.last_scan = image_np

    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(state.device)

    with torch.no_grad():
        logits = model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

    heatmap: Optional[np.ndarray] = None
    heatmap_error: Optional[str] = None
    gradcam = GradCAM(model)
    try:
        heatmap, _, _ = gradcam.generate_heatmap(image_tensor, target_class=predicted_class)
    except GradCAMUnavailableError as exc:
        heatmap_error = str(exc)
    finally:
        gradcam.remove_hooks()

    findings = get_explanation_text(predicted_class, confidence, probabilities=probs.tolist(), heatmap=heatmap)

    true_name = LABEL_NAMES.get(true_class, "Uploaded scan (no reference label)" if true_class == -1 else "Unknown")

    return DiagnoseResponse(
        predicted_class=predicted_class,
        predicted_name=LABEL_NAMES.get(predicted_class, "Unknown"),
        true_class=true_class,
        true_name=true_name,
        confidence=round(confidence * 100.0, 2),
        probabilities=[round(float(p) * 100.0, 2) for p in probs],
        findings=findings,
        raw_image=image_np.tolist(),
        heatmap=heatmap.tolist() if heatmap is not None else None,
        heatmap_available=heatmap is not None,
        heatmap_error=heatmap_error,
        image_source=image_source,
    )


@app.get("/api/cdss/rag-similar", response_model=RagSimilarResponse)
async def get_similar_rag_cases(top_k: int = Query(default=2, ge=1, le=10)):
    """Retrieve the most similar cases from the loaded reference cohort for the last diagnosed scan."""
    server = require_trained_server()
    if state.last_scan is None:
        raise HTTPException(status_code=409, detail="Run a diagnosis first; case retrieval embeds the last diagnosed scan.")

    model = server.get_model()
    model.eval()

    if state.rag_matcher is None:
        state.rag_matcher = build_rag_matcher(server)

    image_tensor = (
        torch.tensor(state.last_scan, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(state.device)
    )

    with torch.no_grad():
        logits = model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        query_class = int(torch.argmax(logits, dim=1).item())

    query_embedding = extract_embedding(model, image_tensor)
    similar = state.rag_matcher.find_similar(query_embedding, top_k=top_k)

    cases = [
        RagTwinCase(
            case_id=s["case_id"],
            label_id=s["label"],
            label_name=LABEL_NAMES.get(s["label"], "Unknown"),
            similarity=round(s["similarity"] * 100.0, 1),
            history=s["clinical_notes"],
        )
        for s in similar
    ]

    return RagSimilarResponse(query_class=query_class, matched_cases=cases)


@app.get("/api/cdss/voice")
async def stream_voice_briefing(
    text: str = Query(..., description="Text to synthesize"),
    lang: str = Query(default="en"),
):
    """Generate audio MP3 briefing using gTTS (offloaded to thread pool).

    gTTS makes blocking HTTP requests to Google TTS API.
    Running it via async worker keeps the event loop free so
    Render / uvicorn can still serve other requests during synthesis.
    """
    try:
        audio_bytes = await get_or_create_audio(text, lang)
        if not audio_bytes:
            raise HTTPException(status_code=503, detail="TTS returned empty audio")
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="briefing.mp3"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {exc}")


@app.get("/api/cdss/report-pdf")
async def download_diagnostic_pdf(
    patient_id: str = Query(..., min_length=1),
    predicted_class: int = Query(..., ge=0, le=2),
    confidence: float = Query(..., ge=0.0, le=100.0),
    findings: str = Query(..., min_length=1),
    lang: str = Query(default="en"),
):
    """Generate the medical PDF report for the last diagnosed scan."""
    if state.last_scan is None:
        raise HTTPException(status_code=409, detail="No diagnosed scan is available. Run a diagnosis before exporting a report.")

    p_name = LABEL_NAMES.get(predicted_class, "Unknown")

    try:
        pdf_bytes = generate_medical_report(
            patient_id=patient_id,
            diagnosis=p_name,
            confidence=confidence,
            explanation=findings,
            original_image=state.last_scan,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="FedXRay_Report_{patient_id}.pdf"'},
    )
