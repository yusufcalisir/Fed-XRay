"""End-to-End Full Lifecycle Federated Medical AI Pipeline Test."""

import torch

from src.fed_xray.data.real_world import StrategyEDatasetEcosystem
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.privacy import PatientLevelDPAccountant
from src.fed_xray.cdss.xai import GradCAM
from src.fed_xray.cdss.similarity import FederatedRAGCaseMatcher, extract_embedding
from src.fed_xray.cdss.report import generate_medical_report


def test_full_federated_lifecycle_pipeline():
    """Verify entire workflow: Ingestion -> FedDyn Training -> Grad-CAM -> RAG -> PDF Report."""
    device = torch.device("cpu")
    ecosystem = StrategyEDatasetEcosystem(seed=42)

    # 1. Ingestion: Strategy E Scenario G (Combined Skew)
    records = ecosystem.generate_synthetic_realworld_cohort(
        dataset_name="ISIC_2019",
        num_patients=80,
        samples_per_patient=2,
        num_classes=3,
    )
    train_recs, _, test_recs = ecosystem.leak_free_patient_split(records, seed=42)
    partitions = ecosystem.partition_into_scenarios(train_recs, num_clients=3, scenario="G")

    dataloaders = [ecosystem.records_to_dataloader(p, batch_size=16) for p in partitions]
    test_images = torch.stack([r.image_tensor for r in test_recs])
    test_labels = torch.tensor([r.label for r in test_recs], dtype=torch.long)

    # 2. Server & Clients
    server = CentralServer(
        device=device,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=8,
        deep_layers_only=True,
    )

    clients = [
        HospitalClient(
            client_id=i,
            dataloader=dataloaders[i],
            device=device,
            learning_rate=0.005,
            local_epochs=1,
            model_type="vit_tiny",
            peft_mode="ffa_lora",
            lora_r=8,
            deep_layers_only=True,
        )
        for i in range(3)
    ]

    dp_accountant = PatientLevelDPAccountant(target_delta=1e-5)

    # 3. Multi-Round FedDyn Training
    for r in range(1, 3):
        agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
            server=server,
            clients=clients,
            round_num=r,
            total_rounds=2,
            test_images=test_images,
            test_labels=test_labels,
            algorithm="FedDyn",
            loss_fn_name="DAFL",
            feddyn_alpha=0.01,
        )
        dp_accountant.step(q=0.1, sigma=1.5)
        assert "loss" in agg_metrics
        assert test_metrics is not None

    # 4. CDSS Diagnostic Inference & Grad-CAM
    global_model = server.get_model()
    global_model.eval()

    sample_scan = test_images[0:1]
    with torch.no_grad():
        logits = global_model(sample_scan)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(probs.argmax())

    gradcam = GradCAM(global_model)
    heatmap, _, _ = gradcam.generate_heatmap(sample_scan, target_class=pred_class)
    gradcam.remove_hooks()

    assert heatmap.shape in ((28, 28), (224, 224))
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    # 5. Federated RAG Digital Twin Retrieval
    rag = FederatedRAGCaseMatcher(reference_records=train_recs, model=global_model)
    query_vec = extract_embedding(global_model, sample_scan)
    twins = rag.find_similar(query_vec, top_k=2)
    assert len(twins) == 2
    assert "similarity" in twins[0]
    assert twins[0]["case_id"].startswith("PAT_")

    # 6. Medical Report PDF Generation
    pdf_bytes = generate_medical_report(
        patient_id="PX-LIFECYCLE-101",
        diagnosis="Pneumonia (Consolidation)",
        confidence=94.5,
        explanation="Consistent consolidation observed across right middle lobe with air bronchograms.",
        original_image=test_images[0, 0].numpy(),
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000


def test_full_federated_lifecycle_strategy_e_224():
    """Verify entire workflow on Strategy E disk manifest at native 224x224 with TenSEAL CKKS."""
    device = torch.device("cpu")
    ecosystem = StrategyEDatasetEcosystem(seed=42)

    # 1. Ingestion: Strategy E Disk Manifest at native 224x224
    records = ecosystem.load_from_disk_manifest(
        manifest_path="data/manifests/strategy_e_manifest.csv",
        images_dir="data",
        target_size=(224, 224),
    )
    train_recs, _, test_recs = ecosystem.leak_free_patient_split(records, seed=42)
    partitions = ecosystem.partition_into_scenarios(train_recs, num_clients=4, scenario="A")

    dataloaders = [ecosystem.records_to_dataloader(p, batch_size=16) for p in partitions]
    test_images = torch.stack([r.image_tensor for r in test_recs])
    test_labels = torch.tensor([r.label for r in test_recs], dtype=torch.long)

    # 2. Server & Clients parameterized for native 224x224
    server = CentralServer(
        device=device,
        model_type="vit_tiny",
        peft_mode="ffa_lora",
        lora_r=8,
        img_size=224,
        deep_layers_only=True,
        use_ckks=True,
    )

    clients = [
        HospitalClient(
            client_id=i,
            dataloader=dataloaders[i],
            device=device,
            learning_rate=0.001,
            local_epochs=1,
            model_type="vit_tiny",
            peft_mode="ffa_lora",
            lora_r=8,
            img_size=224,
            deep_layers_only=True,
        )
        for i in range(4)
    ]

    # 3. One full round with TenSEAL CKKS Homomorphic Encryption
    agg_metrics, client_metrics, test_metrics, sec_report = run_federated_round(
        server=server,
        clients=clients,
        round_num=1,
        total_rounds=1,
        test_images=test_images,
        test_labels=test_labels,
        algorithm="FedDyn",
        loss_fn_name="DAFL",
        use_ckks=True,
    )
    assert "loss" in agg_metrics
    assert test_metrics is not None

    # 4. Native 224x224 Grad-CAM
    global_model = server.get_model()
    sample_scan = test_images[0:1]
    gradcam = GradCAM(global_model)
    heatmap, pred_class, conf = gradcam.generate_heatmap(sample_scan)
    gradcam.remove_hooks()

    assert heatmap.shape == (224, 224), f"Expected 224x224 heatmap, got {heatmap.shape}"

    # 5. Evidence-grounded Federated RAG Case Retrieval
    rag = FederatedRAGCaseMatcher(reference_records=train_recs[:30], model=global_model)
    query_vec = extract_embedding(global_model, sample_scan)
    twins = rag.find_similar(query_vec, top_k=2)
    assert len(twins) == 2
    assert twins[0]["case_id"].startswith("PAT_")

    # 6. PDF Report with genuine 224x224 scan and heatmap
    pdf_bytes = generate_medical_report(
        patient_id="PT-STRAT-E-001",
        diagnosis=f"Clinical Diagnosis (Class {pred_class})",
        confidence=conf * 100.0,
        explanation="Strategy E real clinical validation with verified biopsy twins.",
        original_image=sample_scan[0, 0].numpy(),
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000
