"""
Fed-XRay FastAPI Async Backend Application
==========================================
High-performance REST & Server-Sent Events (SSE) streaming API for the
decoupled Next.js 14 frontend.
"""

import io
import time
import copy
import json
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, List, Optional, Any

# Local Fed-XRay engine imports
from src.fed_xray.data.generator import MedicalDataGenerator, create_hospital_dataloaders, get_distribution_info, create_global_test_set
from src.fed_xray.models.cnn import create_model, count_parameters
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.cdss.xai import GradCAM, create_overlay, get_explanation_text
from src.fed_xray.cdss.similarity import HistoricalCaseBank, extract_embedding, LABEL_NAMES, LABEL_COLORS
from src.fed_xray.cdss.voice import get_or_create_audio
from src.fed_xray.cdss.report import generate_medical_report, get_diagnosis_explanation
from src.fed_xray.cdss.i18n import get_text, get_all_texts

from api.schemas import (
    CohortGenerateRequest,
    CohortGenerateResponse,
    HospitalCohortInfo,
    TrainFLRequest,
    DiagnoseRequest,
    DiagnoseResponse,
    RagSimilarResponse,
    RagTwinCase
)

app = FastAPI(
    title="Fed-XRay Clinical AI API",
    description="Decoupled Federated Learning & Multimodal CDSS API",
    version="2.0.0"
)

# CORS Middleware for Next.js & Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows localhost:3000, Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Application State
class AppState:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloaders = None
        self.global_test_set = None
        self.hospital_samples = {}
        self.server = None
        self.trained_weights = None
        self.model_trained = False
        self.case_bank = None
        self.last_scan = None
        self.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': [], 'test_loss': [],
            'blocked_count': 0
        }

state = AppState()

INSTITUTION_NAMES = [
    "Metropolitan General (Pulmonology Hub)",
    "St. Jude Infectious Disease Center",
    "Community Memorial Health Network",
    "University Medical Academy",
    "St. Mary Pulmonary Screening Clinic",
    "Regional Trauma & ICU Center",
    "Coastline Diagnostic Institute",
    "Mount Sinai Respiratory Lab"
]


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fed-XRay Backend Engine",
        "device": str(state.device),
        "model_trained": state.model_trained
    }


@app.get("/api/state")
async def get_state():
    """Get current consortium and model status."""
    return {
        "model_trained": state.model_trained,
        "hospitals_loaded": len(state.hospital_samples),
        "has_test_set": state.global_test_set is not None,
        "history_rounds": len(state.training_history['round'])
    }


@app.post("/api/cohorts/generate", response_model=CohortGenerateResponse)
async def generate_cohorts(req: CohortGenerateRequest):
    """Synthesize multi-hospital non-IID patient cohorts."""
    generator = MedicalDataGenerator()
    state.hospital_samples = {}
    hospitals_info: List[HospitalCohortInfo] = []
    
    for h in range(req.num_hospitals):
        dist = get_distribution_info(h, req.num_hospitals)
        images, labels = generator.create_hospital_data(
            n_samples=req.samples_per_hospital,
            distribution=dist,
            hospital_id=h
        )
        
        sample_imgs = [img.tolist() for img in images[:9]]
        sample_lbls = [int(lbl) for lbl in labels[:9]]
        
        counts = {
            "normal": int(np.sum(labels == 0)),
            "pneumonia": int(np.sum(labels == 1)),
            "covid": int(np.sum(labels == 2))
        }
        
        inst_name = INSTITUTION_NAMES[h % len(INSTITUTION_NAMES)]
        
        info = HospitalCohortInfo(
            hospital_id=h + 1,
            name=inst_name,
            num_samples=req.samples_per_hospital,
            distribution=[float(x) for x in dist],
            counts=counts,
            sample_images=sample_imgs,
            sample_labels=sample_lbls
        )
        hospitals_info.append(info)
        
        state.hospital_samples[h] = {
            'images': images,
            'labels': labels,
            'distribution': dist
        }
        
    state.dataloaders = create_hospital_dataloaders(
        n_hospitals=req.num_hospitals,
        samples_per_hospital=req.samples_per_hospital,
        batch_size=32
    )
    
    test_images, test_labels = create_global_test_set(n_samples=300, seed=9999)
    state.global_test_set = (test_images, test_labels)
    
    return CohortGenerateResponse(
        success=True,
        num_hospitals=req.num_hospitals,
        total_samples=req.num_hospitals * req.samples_per_hospital,
        hospitals=hospitals_info
    )


@app.get("/api/fl/train-stream")
async def stream_federated_training(
    num_rounds: int = Query(default=5, ge=1, le=20),
    local_epochs: int = Query(default=2, ge=1, le=5),
    learning_rate: float = Query(default=0.0001),
    privacy_noise: float = Query(default=0.01),
    simulate_attack: bool = Query(default=False),
    activate_defense: bool = Query(default=True),
    algorithm: str = Query(default="FedAvg"),
    loss_fn: str = Query(default="CE")
):
    """
    Server-Sent Events (SSE) streaming federated training rounds in real-time.
    """
    if state.dataloaders is None or state.global_test_set is None:
        # Auto-generate if not present
        generator = MedicalDataGenerator()
        state.dataloaders = create_hospital_dataloaders(n_hospitals=4, samples_per_hospital=200, batch_size=32)
        test_images, test_labels = create_global_test_set(n_samples=300, seed=9999)
        state.global_test_set = (test_images, test_labels)
        
    async def event_generator():
        n_hospitals = len(state.dataloaders)
        state.server = CentralServer(device=state.device, privacy_noise=privacy_noise, defense_mode=activate_defense)
        
        clients = []
        for i in range(n_hospitals):
            is_malicious = (simulate_attack and i == 2) # Node 3 is malicious
            clients.append(HospitalClient(
                client_id=i,
                dataloader=state.dataloaders[i],
                device=state.device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                malicious=is_malicious
            ))
            
        test_images, test_labels = state.global_test_set
        
        state.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': [], 'test_loss': [],
            'blocked_count': 0
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
                proto_weight=0.1
            )
            
            t_loss = float(metrics['loss'])
            t_acc = float(metrics['accuracy'] * 100.0)
            
            val_acc = float(test_metrics.accuracy * 100.0) if test_metrics else t_acc
            val_loss = float(test_metrics.loss) if test_metrics else t_loss
            prec = float(test_metrics.precision * 100.0) if test_metrics else 0.0
            rec = float(test_metrics.recall * 100.0) if test_metrics else 0.0
            f1 = float(test_metrics.f1_score * 100.0) if test_metrics else 0.0
            
            blocked_nodes = sec_report.clients_blocked if sec_report else []
            threat_detected = len(blocked_nodes) > 0
            
            state.training_history['round'].append(round_num)
            state.training_history['loss'].append(val_loss)
            state.training_history['accuracy'].append(val_acc)
            state.training_history['test_accuracy'].append(val_acc)
            state.training_history['test_loss'].append(val_loss)
            state.training_history['f1_score'].append(f1)
            
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
                "status": "training" if round_num < num_rounds else "complete"
            }
            
            yield f"data: {json.dumps(data_payload)}\n\n"
            await asyncio.sleep(0.3)
            
        trained_model = state.server.get_model()
        state.trained_weights = copy.deepcopy(trained_model.state_dict())
        state.model_trained = True
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/cdss/diagnose", response_model=DiagnoseResponse)
async def run_diagnosis(req: DiagnoseRequest):
    """Run CDSS inference with Grad-CAM saliency extraction."""
    generator = MedicalDataGenerator(seed=int(time.time()))
    true_label = req.class_index if req.class_index is not None else np.random.randint(0, 3)
    raw_img = generator.generate_synthetic_xray(true_label)
    
    model = create_model()
    if state.trained_weights is not None:
        model.load_state_dict(state.trained_weights)
    model.cpu().eval()
    
    img_tensor = torch.FloatTensor(raw_img).unsqueeze(0).unsqueeze(0)
    torch.set_grad_enabled(True)
    
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence = gradcam.generate_heatmap(img_tensor)
    confidence_pct = float(confidence * 100.0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        
    gradcam.remove_hooks()
    
    class_names = ["Normal (Clear Parenchyma)", "Pneumonia (Focal Consolidation)", "COVID-19 (Ground-Glass Opacities)"]
    findings = get_explanation_text(pred_class, confidence)
    
    state.last_scan = {
        "raw_image": raw_img,
        "heatmap": heatmap,
        "predicted_class": pred_class,
        "confidence": confidence_pct,
        "true_label": true_label
    }
    
    return DiagnoseResponse(
        predicted_class=pred_class,
        predicted_name=class_names[pred_class],
        true_class=true_label,
        true_name=class_names[true_label],
        confidence=round(confidence_pct, 2),
        probabilities=[round(float(p), 4) for p in probs],
        findings=findings,
        raw_image=raw_img.tolist(),
        heatmap=heatmap.tolist()
    )


@app.post("/api/cdss/rag-similar", response_model=RagSimilarResponse)
async def find_similar_cases(req: DiagnoseRequest):
    """Case-Based Reasoning (RAG) Digital Twin Matcher."""
    if state.case_bank is None:
        state.case_bank = HistoricalCaseBank(n_cases=100)
        
    if state.last_scan is None:
        generator = MedicalDataGenerator()
        raw_img = generator.generate_synthetic_xray(1)
    else:
        raw_img = state.last_scan["raw_image"]
        
    model = create_model()
    if state.trained_weights is not None:
        model.load_state_dict(state.trained_weights)
    model.cpu().eval()
    
    img_tensor = torch.FloatTensor(raw_img).unsqueeze(0).unsqueeze(0)
    embedding = extract_embedding(model, img_tensor)
    
    similar_cases = state.case_bank.find_similar(embedding, top_k=2)
    
    matched = []
    for c in similar_cases:
        lbl_name = LABEL_NAMES.get(c['label'], 'Unknown')
        matched.append(RagTwinCase(
            case_id=c['case_id'],
            label_id=c['label'],
            label_name=lbl_name,
            similarity=round(float(c['similarity'] * 100.0), 2),
            history=f"Biopsy verified {lbl_name}. Follow-up completed with favorable response."
        ))
        
    pred_c = state.last_scan["predicted_class"] if state.last_scan else 0
    return RagSimilarResponse(query_class=pred_c, matched_cases=matched)


@app.get("/api/cdss/voice")
async def get_voice_briefing(diagnosis: str = "Normal", confidence: float = 95.0):
    """Generate audio MP3 speech briefing."""
    audio_bytes = get_or_create_audio(diagnosis, confidence)
    if audio_bytes is None:
        raise HTTPException(status_code=500, detail="Voice engine unavailable")
    return Response(content=audio_bytes, media_type="audio/mp3")


@app.get("/api/cdss/report-pdf")
async def download_report_pdf(diagnosis: str = "Pneumonia", confidence: float = 94.2):
    """Generate and stream single-page official A4 medical intelligence PDF report."""
    pat_id = f"PAT-{int(time.time()) % 100000}"
    explanation = get_diagnosis_explanation(diagnosis, confidence)
    
    raw_img = state.last_scan["raw_image"] if state.last_scan else np.zeros((28, 28))
    heatmap = state.last_scan["heatmap"] if state.last_scan else None
    
    pdf_bytes = generate_medical_report(
        patient_id=pat_id,
        diagnosis=diagnosis,
        confidence=confidence,
        explanation=explanation,
        heatmap_image=heatmap,
        original_image=raw_img
    )
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FedXRay_Report_{pat_id}.pdf"}
    )
