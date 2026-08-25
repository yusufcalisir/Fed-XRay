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
import copy
import io
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.fed_xray.data.generator import (
    MedicalDataGenerator,
    create_global_test_set,
    create_hospital_dataloaders,
    get_distribution_info,
)
from src.fed_xray.data.real_world import (
    RealWorldPatientRecord,
    StrategyEDatasetEcosystem,
)
from src.fed_xray.models.cnn import count_parameters, create_model as create_cnn_model
from src.fed_xray.models.vit import create_medical_vit
from src.fed_xray.models.peft import inject_lora_to_model
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.cdss.xai import GradCAM, create_overlay, get_explanation_text
from src.fed_xray.cdss.similarity import (
    FederatedRAGCaseMatcher,
    HistoricalCaseBank,
    extract_embedding,
    LABEL_COLORS,
    LABEL_NAMES,
)
from src.fed_xray.cdss.voice import get_or_create_audio
from src.fed_xray.cdss.report import generate_medical_report, get_diagnosis_explanation
from src.fed_xray.cdss.i18n import get_all_texts, get_text

from api.schemas import (
    CohortGenerateRequest,
    CohortGenerateResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HospitalCohortInfo,
    RagSimilarResponse,
    RagTwinCase,
    RoundTelemetryUpdate,
    TrainFLRequest,
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
        self.dataloaders = None
        self.global_test_set = None
        self.hospital_samples = {}
        self.server: Optional[CentralServer] = None
        self.model_trained = False
        self.model_type = "vit_tiny"
        self.peft_mode = "ffa_lora"
        self.rag_matcher: Optional[FederatedRAGCaseMatcher] = None
        self.last_scan = None
        self.training_history = {
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


state = AppState()

INSTITUTION_NAMES = [
    "BCN-20000 Skin Cancer Hub (Barcelona)",
    "ViDIR Dermatopathology Institute (Vienna)",
    "Queensland Oncology Screening Center",
    "Beth Israel Deaconess Medical Center",
    "NCT National Center for Tumor Diseases",
    "St. Jude Pulmonary & Infection Center",
    "Metropolitan Thoracic Diagnostic Lab",
    "Mount Sinai Radiological Oncology Unit",
]


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
    records = ecosystem.generate_synthetic_realworld_cohort(
        dataset_name=req.dataset_name,
        num_patients=max(60, req.num_hospitals * 25),
        samples_per_patient=max(1, req.samples_per_hospital // 25),
        num_classes=3,
    )

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
    state.dataloaders = []

    for h in range(req.num_hospitals):
        part = client_partitions[h]
        loader = ecosystem.records_to_dataloader(part, batch_size=32, shuffle=True)
        state.dataloaders.append(loader)

        images = np.stack([r.image_tensor.numpy()[0] for r in part])
        labels = np.array([r.label for r in part])

        c0 = int(np.sum(labels == 0))
        c1 = int(np.sum(labels == 1))
        c2 = int(np.sum(labels == 2))
        total_p = max(1, len(labels))

        dist = [c0 / total_p, c1 / total_p, c2 / total_p]
        counts = {"normal": c0, "pneumonia": c1, "covid": c2}

        sample_imgs = [r.image_tensor.numpy()[0].tolist() for r in part[:9]]
        sample_lbls = [int(r.label) for r in part[:9]]

        inst_name = INSTITUTION_NAMES[h % len(INSTITUTION_NAMES)]

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

    # Global hold-out test set from leak-free split
    test_images = torch.stack([r.image_tensor for r in test_recs])
    test_labels = torch.tensor([r.label for r in test_recs], dtype=torch.long)
    state.global_test_set = (test_images, test_labels)

    total_samples = sum(len(p) for p in client_partitions)

    return CohortGenerateResponse(
        success=True,
        num_hospitals=req.num_hospitals,
        total_samples=total_samples,
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
):
    """Server-Sent Events (SSE) streaming federated ViT training in real time."""
    if state.dataloaders is None or state.global_test_set is None:
        ecosystem = StrategyEDatasetEcosystem(seed=42)
        records = ecosystem.generate_synthetic_realworld_cohort(num_patients=100, samples_per_patient=3)
        train_recs, _, test_recs = ecosystem.leak_free_patient_split(records)
        partitions = ecosystem.partition_into_scenarios(train_recs, num_clients=4, scenario="A")
        state.dataloaders = [ecosystem.records_to_dataloader(p, batch_size=32) for p in partitions]
        state.global_test_set = (
            torch.stack([r.image_tensor for r in test_recs]),
            torch.tensor([r.label for r in test_recs], dtype=torch.long),
        )

    actual_peft = None if peft_mode.lower() in ("none", "false", "off") else peft_mode
    state.model_type = model_type
    state.peft_mode = peft_mode

    async def event_generator():
        n_hospitals = len(state.dataloaders)
        state.server = CentralServer(
            device=state.device,
            privacy_noise=privacy_noise,
            defense_mode=activate_defense,
            model_type=model_type,
            peft_mode=actual_peft,
            lora_r=8,
            lora_alpha=16.0,
            deep_layers_only=True,
        )

        clients = []
        for i in range(n_hospitals):
            is_malicious = simulate_attack and i == 2
            clients.append(
                HospitalClient(
                    client_id=i,
                    dataloader=state.dataloaders[i],
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

        test_images, test_labels = state.global_test_set

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
                "blocked_nodes": [int(n + 1) for n in blocked_nodes],
                "status": "training" if round_num < num_rounds else "complete",
                "model_type": model_type,
                "peft_mode": peft_mode,
            }

            yield f"data: {json.dumps(data_payload)}\n\n"
            await asyncio.sleep(0.05)

        state.model_trained = True
        state.rag_matcher = FederatedRAGCaseMatcher(n_cases=100, embedding_dim=64)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cdss/diagnose", response_model=DiagnoseResponse)
async def perform_diagnosis(req: DiagnoseRequest):
    """Diagnose patient scan and generate Grad-CAM visual explanation."""
    if state.server is None:
        state.server = CentralServer(
            device=state.device,
            model_type=state.model_type,
            peft_mode=state.peft_mode,
        )

    model = state.server.get_model()
    model.eval()

    generator = MedicalDataGenerator(seed=int(time.time() * 1000) % 100000)
    true_class = (
        req.class_index
        if req.class_index is not None and req.class_index in (0, 1, 2)
        else np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
    )

    image_np = generator.generate_synthetic_xray(label=true_class, apply_augmentation=True)
    state.last_scan = image_np

    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(state.device)

    with torch.no_grad():
        logits = model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

    gradcam = GradCAM(model)
    heatmap, _, _ = gradcam.generate_heatmap(image_tensor, target_class=predicted_class)
    gradcam.remove_hooks()

    findings = get_explanation_text(predicted_class, confidence)

    return DiagnoseResponse(
        predicted_class=predicted_class,
        predicted_name=LABEL_NAMES.get(predicted_class, "Unknown"),
        true_class=true_class,
        true_name=LABEL_NAMES.get(true_class, "Unknown"),
        confidence=round(confidence * 100.0, 2),
        probabilities=[round(float(p) * 100.0, 2) for p in probs],
        findings=findings,
        raw_image=image_np.tolist(),
        heatmap=heatmap.tolist(),
    )


@app.get("/api/cdss/rag-similar", response_model=RagSimilarResponse)
async def get_similar_rag_cases(query_class: int = Query(default=1)):
    """Retrieve top-K verified biopsy-confirmed digital twin cases."""
    if state.rag_matcher is None:
        state.rag_matcher = FederatedRAGCaseMatcher(n_cases=100, embedding_dim=64)

    target_c = query_class if query_class in (0, 1, 2) else 1

    sample_query = np.zeros(64, dtype=np.float32)
    if target_c == 0:
        sample_query[:20] = 1.0
    elif target_c == 1:
        sample_query[20:40] = 1.0
    else:
        sample_query[40:] = 1.0

    sample_query += np.random.randn(64).astype(np.float32) * 0.1
    sample_query = sample_query / np.linalg.norm(sample_query)

    similar = state.rag_matcher.find_similar(sample_query, top_k=2)

    cases = []
    for s in similar:
        cases.append(
            RagTwinCase(
                case_id=s["case_id"],
                label_id=s["label"],
                label_name=LABEL_NAMES.get(s["label"], "Unknown"),
                similarity=round(s["similarity"] * 100.0, 1),
                history=s["biopsy_finding"],
            )
        )

    return RagSimilarResponse(query_class=target_c, matched_cases=cases)


@app.get("/api/cdss/voice")
async def stream_voice_briefing(text: str = Query(...), lang: str = Query(default="en")):
    """Generate audio MP3 briefing using edge-tts / gTTS."""
    audio_bytes = await get_or_create_audio(text, lang)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="briefing.mp3"'},
    )


@app.get("/api/cdss/report-pdf")
async def download_diagnostic_pdf(
    patient_id: str = Query(default="PX-9042"),
    predicted_class: int = Query(default=1),
    confidence: float = Query(default=92.4),
    findings: str = Query(default="Consolidation in right lower lobe."),
    lang: str = Query(default="en"),
):
    """Generate medical PDF report download."""
    p_name = LABEL_NAMES.get(predicted_class, "Normal")
    pdf_bytes = generate_medical_report(
        patient_id=patient_id,
        diagnosis=p_name,
        confidence=confidence,
        explanation=findings,
        original_image=state.last_scan,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="FedXRay_Report_{patient_id}.pdf"'},
    )
