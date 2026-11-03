# Fed-XRay: Federated Vision Transformers for Medical Oncology AI

**Privacy-Preserving, Personalized, and Multimodal Federated Learning for Radiological and Pathology Imaging**

> [!CAUTION]
> **Research Prototype - Not for Clinical Use.**
> This system is a research prototype. It has not been validated or approved for clinical deployment. Outputs must not be used for diagnostic or treatment decisions. All results are experimental and intended for academic and engineering study only.

<div align="center">
<a href="https://yusuf-cancerfedxlearning.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit_Legacy_Demo-Access_Platform-emerald?style=for-the-badge&logo=streamlit&logoColor=white" alt="Legacy Demo">
</a>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Next.js-14.2+-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Privacy-DP_%2B_CKKS_(Simulated)_%2B_SecAgg%2B-indigo" alt="Privacy">
  <img src="https://img.shields.io/badge/PEFT-FFA--LoRA_%7C_FedSA--LoRA_%7C_MedSAM_%7C_FedMedCLIP-purple" alt="PEFT">
  <img src="https://img.shields.io/badge/Flower_FL-Supported-green?logo=flower" alt="Flower FL">
  <img src="https://img.shields.io/badge/Localization-EN%20%7C%20TR-blue" alt="Bilingual">
  <img src="https://img.shields.io/badge/Tests-326_Passed_(246_Backend_%2B_80_Frontend)-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Table of Contents

- [1. Research Prototype Disclaimer](#1-research-prototype-disclaimer)
- [2. Project Overview and Mission](#2-project-overview-and-mission)
- [3. Implementation Status](#3-implementation-status)
- [4. Architecture and Protocol](#4-architecture-and-protocol)
  - [4.1 End-to-End Workflow Diagram](#41-end-to-end-workflow-diagram)
  - [4.2 CKKS and Byzantine Detection: Protocol Phases](#42-ckks-and-byzantine-detection-protocol-phases)
  - [4.3 CKKS Key Model Limitations](#43-ckks-key-model-limitations)
- [5. Actually Implemented Models](#5-actually-implemented-models)
- [6. PEFT Methods and Bilinear Discordance Elimination](#6-peft-methods-and-bilinear-discordance-elimination)
- [7. Drift-Resilient Federated Optimizers](#7-drift-resilient-federated-optimizers)
- [8. Foundation-Anchored Prototype Metric Learning and Imbalance Losses](#8-foundation-anchored-prototype-metric-learning-and-imbalance-losses)
- [9. Privacy Architecture](#9-privacy-architecture)
- [10. Strategy E Datasets and Licenses](#10-strategy-e-datasets-and-licenses)
  - [10.1 Dataset Sources](#101-dataset-sources)
  - [10.2 Non-IID Heterogeneity Scenarios (A to G)](#102-non-iid-heterogeneity-scenarios-a-to-g)
  - [10.3 Leak-Free Patient-Level Cohort Isolation](#103-leak-free-patient-level-cohort-isolation)
- [11. Benchmark Claims](#11-benchmark-claims)
- [12. Clinical Decision Support System (CDSS)](#12-clinical-decision-support-system-cdss)
- [13. Multi-Platform Interfaces](#13-multi-platform-interfaces)
- [14. Flower Protocol Federation Bridge](#14-flower-protocol-federation-bridge)
- [15. REST and Streaming API Reference](#15-rest-and-streaming-api-reference)
- [16. Test Suite Matrix (326 Tests)](#16-test-suite-matrix-326-tests)
- [17. Repository Directory Structure](#17-repository-directory-structure)
- [18. Installation and Quick Start](#18-installation-and-quick-start)
- [19. Scientific Anchors and Citations](#19-scientific-anchors-and-citations)

---

## 1. Research Prototype Disclaimer

This repository is an **academic research prototype** developed for studying federated learning techniques applied to medical imaging. The following disclaimers apply throughout:

- **No clinical validation:** No component of this system has been validated against clinical outcomes or approved by any medical regulatory authority (FDA, CE, MHRA, or equivalent).
- **Not for diagnostic use:** Model outputs, saliency maps, and RAG-retrieved cases are research artifacts, not clinical evidence. They must not inform diagnostic or treatment decisions.
- **Simulated privacy:** CKKS Homomorphic Encryption is implemented via TenSEAL in an aggregation-only simulation mode. SecAgg+ uses in-process Python cryptography, not a real multi-party network protocol. See [Section 9](#9-privacy-architecture) for full details.
- **Dataset references:** The hospital node names BCN-20000, ViDIR, Queensland Oncology Center, and Beth Israel Deaconess are references to the public datasets' original contributing institutions (ISIC 2019, MIMIC-CXR). They do not represent active institutional partnerships or data access agreements.
- **Benchmark values:** AUC figures cited in benchmarks are literature reference targets, not measurements from this codebase's demo environment. See [Section 11](#11-benchmark-claims).

---

## 2. Project Overview and Mission

Collaborative AI in radiological oncology and computational pathology is constrained by institutional data governance, cross-scanner covariate shifts, extreme class prevalence heterogeneity, and gigapixel transmission bottlenecks. Centralizing raw patient imaging data introduces re-identification hazards, violates HIPAA/GDPR/EHDS directives, and incurs multi-gigabyte communication overheads.

Naive federated learning approaches that train large Vision Transformers (ViTs) from scratch diverge under non-convex loss surfaces, suffer from client drift, and require up to 2.43 GB of uncompressed weight transfers per hospital per round.

**Fed-XRay** addresses these challenges through:

1. **Medical Foundation-Backed PEFT:** Adapting domain-specific representations via factor-wise asymmetric LoRA variants (FFA-LoRA, FedSA-LoRA, FedAS-LoRA), reducing uplink bandwidth by greater than 99.8% (under 2.36 MB/round).
2. **Option J Privacy Composition:** Combining SecAgg+ ECDH masking, TenSEAL CKKS ciphertext aggregation (simulation), and patient-level Gaussian Differential Privacy with Rényi DP accounting.
3. **Strategy E Real-World Benchmarks:** Partitioning genuine multi-center clinical cohorts (ISIC 2019, NCT-CRC-HE, MIMIC-CXR) under strict patient-level isolation across 7 controlled non-IID scenarios.
4. **Evidence-Grounded CDSS:** Indexing backbone embeddings into an HNSW vector index for retrieval of biopsy-confirmed reference cases, paired with Grad-CAM saliency, PDF diagnostic reports, and bilingual audio dictation.

> [!NOTE]
> **Multi-task federation note:** This demo combines dermoscopy (ISIC 2019), histopathology (NCT-CRC-HE), and chest X-ray (MIMIC-CXR) under a single global model. This constitutes a multi-task federated setting, not a standard single-task Non-IID FL setup. Cross-modality global model averaging is an active research challenge. The current implementation uses a shared backbone with task-specific classification heads. Performance under this setup is not directly comparable to single-modality federated learning benchmarks.

---

## 3. Implementation Status

The table below distinguishes what is actually coded and exercised in this repository, what is approximated or simulated, and what remains an active research target.

| Component | Status | Notes |
| :--- | :--- | :--- |
| ViT-Tiny (192-dim, 5.7M params) | **Implemented** | Fully integrated into training, CDSS, and automated tests |
| ViT-Base (768-dim, 85M params) | **Implemented** | Integrated with FFA-LoRA and FedSA-LoRA adapter injection |
| Swin UNETR Radiology Backbone | **Implemented** | Hierarchical Shifted Window Attention, patch merging, 2D/3D volumetric support, official Swin-T or MONAI weights |
| UNI Foundation Backbone | **Implemented** | Transfer-adapted pathology features, SHA-256 weight fingerprinting, strict checkpoint verification |
| MedSAM Client Adapter (FCA) | **Implemented** | Bottleneck residual scaling integrated into `peft.py` and `client.py` for volumetric segmentation |
| FedMedCLIP Multimodal Module | **Implemented** | Masked FAM + Clinical Text Semantic Anchors + private MLP + mutual KL distillation in `server.py` and `client.py` |
| FedCola Modality Bridge | **Implemented** | Cross-modal projection layers bridging digital pathology and chest X-rays without paired patient data |
| FedDAT Dual-Adapter Cross-Attention | **Implemented** | Dual vision-language bottleneck adapters with mutual cross-attention alignment |
| Text Semantic Anchors | **Implemented** | Frozen `Bio_ClinicalBERT` canonical oncology coordinate system (`TextSemanticAnchor`) |
| XRayClassifier CNN baseline | **Implemented** | Used in benchmarking, Grad-CAM hooks, and comparative evaluation |
| FFA-LoRA adapter injection | **Implemented** | Bilinear error elimination verified in mathematical consistency tests |
| FedSA-LoRA / FedAS-LoRA | **Implemented** | Factor-wise asymmetric adaptation, rank-aware shared subspace routing |
| FlexLoRA SVD reconstruction | **Implemented** | Truncated SVD parameter reconstruction for heterogeneous client budgets |
| FedPerfix parallel plugin | **Implemented** | Attention-decoupled personalization targeting deep attention projections |
| FedDyn optimizer | **Implemented** | Dynamic state tracking and regularization in `server.py` |
| FedOpt / FedAdam server momentum | **Implemented** | Adaptive server-side first and second momentum tracking in `server.py` |
| SCAFFOLD control variates | **Implemented** | Dual client and server state updates correcting drift on PEFT weights |
| MOON contrastive regularization | **Implemented** | Representation-level model-contrastive loss against global and prior local state |
| Patient-level DP clipping | **Implemented** | Honest patient-level gradient clipping preventing gigapixel privacy collapse |
| Rényi DP accounting | **Implemented** | Subsampled Gaussian mechanism composition tracking in `privacy.py` |
| FedProto dispersion-weighted synthesis | **Implemented** | Covariance-trace intra-class weighting in `prototypes.py` |
| DAFL, BSM, LDAM, repel losses | **Implemented** | Dynamic imbalance loss suite with verified numerical stability |
| Grad-CAM (CNN and ViT) | **Implemented** | Multi-layer PyTorch hooks; raises `GradCAMUnavailableError` if gradients fail |
| Structured CDSS XAI (Zero Asterisks) | **Implemented** | Asterisk-free clinical assessment, quadrant spatial localization, bilingual consistency |
| HNSW dynamic-dim RAG retrieval | **Implemented** | Auto-detects 128-dim to 1024-dim latents; matches biopsy-confirmed twins |
| ACID Persistent State Store | **Implemented** | SQLite WAL mode (`PersistentStateStore`) surviving process restarts across 5 relational tables |
| Network Routing & Tunnel Resilience | **Implemented** | Dynamic ephemeral tunnel detection, dead domain blacklisting, automatic loopback fallback |
| Browser Runtime Shield | **Implemented** | Client-side interception of extension collisions (`useCache`, `content.js`), port drop isolation |
| React Subtree Resilience | **Implemented** | Next.js `app/error.tsx`, `app/global-error.tsx`, and `ClinicalErrorBoundary` with bilingual recovery |
| PDF diagnostic report engine | **Implemented** | `fpdf2`-based with embedded scan, Grad-CAM, RAG twins, and SHA-256 hash |
| Bilingual voice dictation (EN/TR) | **Implemented** | Asynchronous `gTTS` speech synthesis with persistent disk cache |
| Byzantine detection (cosine + deviance) | **Implemented (plaintext phase)** | Runs before simulated CKKS encryption; see Section 4.2 |
| SecAgg+ ECDH zero-sum masking | **Simulated** | In-process Python simulation, not a real multi-party network protocol |
| CKKS HE aggregation (TenSEAL) | **Simulated** | Single server-held key, aggregation-only depth; see Section 4.3 |
| Flower protocol bridge | **Implemented** | `NumPyClient` wrapping with `InProcessClientProxy` and `StrategyEHospitalStrategy` |
| Virchow2 (ViT-H/14, 632M params) | **Planned** | Referenced as target; not loaded or downloaded |
| Prov-GigaPath (1.3B params) | **Planned** | Referenced as target; not loaded or downloaded |
| True threshold multi-party CKKS | **Planned** | Requires distributed key generation; active research target |

---

## 4. Architecture and Protocol

### 4.1 End-to-End Workflow Diagram

```mermaid
flowchart TD
    subgraph HospitalNodes["Distributed Hospital Consortium (Edge Clients)"]
        H1["Hospital 1 (BCN-20000 ref.)\nLocal Scans (Patient-Level Split)\nFrozen Backbone + Local LoRA B"]
        H2["Hospital 2 (ViDIR ref.)\nLocal Scans (Patient-Level Split)\nFrozen Backbone + Local LoRA B"]
        H3["Hospital 3 (Queensland ref.)\nLocal Scans (Patient-Level Split)\nFrozen Backbone + Local LoRA B"]
        H4["Hospital 4 (BIDMC ref.)\nLocal Scans (Patient-Level Split)\nFrozen Backbone + Local LoRA B"]
    end

    subgraph ClientPrivacy["Client-Side Privacy (Phase 1 - Plaintext)"]
        DP["Patient-Level L2 DP Clipping\nRényi DP Accounting (eps<=2.0)"]
        Byz["Byzantine Self-Check\nLocal Gradient Norm Validation"]
    end

    subgraph ServerByz["Server Byzantine Detection (Phase 2 - Plaintext)"]
        ByzServer["Cosine Distance Check\nLoss Deviance Validation\nAdversary Quarantine"]
    end

    subgraph CipherPhase["Aggregation Privacy (Phase 3 - Simulated Ciphertext)"]
        SecAgg["SecAgg+ Zero-Sum ECDH Masking\n(In-Process Simulation)"]
        CKKS["TenSEAL CKKS Ciphertext Addition\n(Single Server Key - Simulated)"]
    end

    subgraph ServerCore["Central Server Aggregation"]
        Opt["FedDyn / FedAdam / SCAFFOLD\nServer-Side Optimization"]
        Proto["FedProto Covariance-Trace Synthesis"]
    end

    subgraph CDSSStudio["Clinical Decision Support (CDSS)"]
        RAG["HNSW RAG Case Bank\n128-dim to 768-dim Indexing"]
        XAI["Grad-CAM Saliency Engine"]
        Report["PDF Report + Voice Dictation"]
    end

    HospitalNodes --> DP
    DP --> Byz
    Byz --> ByzServer
    ByzServer --> SecAgg
    SecAgg --> CKKS
    CKKS --> Opt
    Opt --> Proto
    Proto -->|Broadcast Global Parameters| HospitalNodes
    Opt --> CDSSStudio
```

### 4.2 CKKS and Byzantine Detection: Protocol Phases

> [!IMPORTANT]
> **Honest Architecture Note - CKKS vs Byzantine Detection Conflict**
>
> The two privacy mechanisms operate at different protocol phases. They are not simultaneously active on the same data:
>
> - **Phase 1 (Client-side, plaintext):** Each hospital client applies patient-level DP gradient clipping, performs local gradient norm self-checks, and then submits updates. The server runs Byzantine anomaly detection (cosine distance and loss deviance scoring) on these plaintext updates BEFORE any encryption is applied. The server can read plaintext updates at this stage.
>
> - **Phase 2 (Server aggregation, simulated ciphertext):** After Byzantine filtering, surviving updates are masked with SecAgg+ zero-sum vectors and passed through TenSEAL CKKS ciphertext addition. In practice, because Byzantine detection already completed in plaintext, the CKKS phase provides privacy for the aggregation step only.
>
> - **Fundamental constraint:** The server cannot simultaneously maintain zero-knowledge CKKS ciphertext privacy AND perform coordinate-wise trimmed mean ranking or hold-out validation loss deviance checking. These operations require reading plaintext update coordinates. The current implementation correctly separates these phases, but this means the server observes plaintext updates during Byzantine detection. True zero-knowledge Byzantine defense would require threshold HE or MPC techniques and is a research target.

When ciphertext masking is active, the server reports:
```json
{ "threat_detection_status": "unverifiable_under_ciphertext_masking" }
```
This is the honest status for the pure CKKS aggregation phase.

### 4.3 CKKS Key Model Limitations

> [!WARNING]
> **Single Server Key Model - Not Multi-Party Threshold CKKS**
>
> The current TenSEAL CKKS implementation uses a SINGLE server-held key pair (`poly_modulus_degree=8192`, `scale_bits=40`). This means:
> - The server generates, holds, and can use the decryption key.
> - The server CAN decrypt individual encrypted updates if it chooses to do so.
> - This does NOT provide the zero-knowledge aggregation guarantee of threshold multi-party CKKS (where no single party holds the full decryption key).
>
> True threshold/multi-party CKKS requires distributed key generation protocols and is a planned research target. The current implementation demonstrates ciphertext addition mechanics and lays the groundwork for future multi-party integration.

---

## 5. Actually Implemented Models

The following neural architectures are implemented, loaded, and integrated end-to-end in training, inference, CDSS, and testing:

| Model | Parameters | Embedding Dim | Implementation File | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ViT-Tiny** | 5.7M | 192 | `src/fed_xray/models/vit.py` | Fully implemented and trained |
| **ViT-Base** | 85M | 768 | `src/fed_xray/models/vit.py` | Implemented with FFA-LoRA and FedSA-LoRA |
| **Swin UNETR Radiology Backbone** | ~28M to 62M | 768 / 384 | `src/fed_xray/models/foundation.py` | Hierarchical Shifted Window Attention, patch merging, 2D/3D volumetric support, official Swin-T or MONAI weights |
| **UNI Foundation Backbone** | ~304M | 1024 / 512 | `src/fed_xray/models/foundation.py` | Pathology foundation model, transfer features, SHA-256 weight hash, strict checkpoint loading |
| **FedMedCLIP Multimodal Model** | ~10M to 95M | 512 | `src/fed_xray/models/foundation.py` | Couples vision backbone with FAM, frozen clinical text anchors, and private MLP |
| **XRayClassifier (CNN)** | ~2M | 128 | `src/fed_xray/models/cnn.py` | Fully implemented convolutional baseline |

The following are referenced as **planned high-capacity foundation models** for multi-billion parameter Whole Slide Imaging (WSI) deployments:

| Target Model | Params | Role in Literature | Status |
| :--- | :--- | :--- | :--- |
| Virchow2 (ViT-H/14) | ~632M | Pan-cancer pathology foundation encoder | Planned target |
| Prov-GigaPath | ~1.3B | Whole-slide gigapixel pathology transformer | Planned target |

> [!NOTE]
> **Weight Authenticity and Checkpoint Integrity (Rule 12):** All foundation backbones compute SHA-256 cryptographic digests of projection weight matrices and enforce honest exception raising on missing or corrupted weights. Silent swallowing of checkpoint load failures (`except Exception: pass`) is prohibited.
>
> **Benchmark table clarification:** The communication economics table in Section 11 references "Full Foundation ViT-H/14" at 2,430 MB and "Full ViT-B" at approximately 340 MB. These sizes correspond to ViT-H/14 (~632M params) and ViT-B/16 (~85M params) weight tensors respectively, used as theoretical baselines. The implemented ViT-Tiny (5.7M), ViT-Base (85M), and Swin UNETR models are used in actual runs. Communication savings figures for PEFT methods are computed relative to the full ViT-B baseline.

---

## 6. PEFT Methods and Bilinear Discordance Elimination

In standard federated low-rank adaptation, each client decomposes weight adjustments as $\Delta W_k = B_k A_k$. Central server parameter averaging introduces the **Bilinear LoRA Aggregation Error**:

$$
\bar{\Delta W}_{\text{effective}} = \left(\sum_{k=1}^K p_k B_k\right)\left(\sum_{k=1}^K p_k A_k\right) = \bar{B}\bar{A} \neq \sum_{k=1}^K p_k (B_k A_k) = \bar{\Delta W}_{\text{ideal}}
$$

Fed-XRay deploys factor-wise asymmetric adaptation to eliminate this distortion, alongside specialized multimodal adapters:

| Method | Matrix A Protocol | Matrix B Protocol | Aggregation Formula | Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **FFA-LoRA** | Globally Frozen ($A_k \equiv A_0$) | Locally Trained and Aggregated | $\bar{B}A_0 \equiv \sum p_k B_k A_0$ | Eliminates bilinear error identically; compatible with multiplication-free CKKS (Depth=0). |
| **FedSA-LoRA** | Locally Trained and Aggregated ($\bar{A}$) | Locally Retained (Private $B_k$) | Global $\bar{A}$, Private $B_k$ | Aggregates global visual input subspace; keeps staining/scanner adaptation private. |
| **FedAS-LoRA** | Adaptive Routing | Adaptive Routing | Rank-Aware Shared Subspace (RSS) | Dynamically toggles Share-A/Local-B (covariate shift) vs Share-B/Local-A (label skew). |
| **FlexLoRA** | Locally Trained | Locally Trained | $U_r \Sigma_r V_r^\top = \sum p_k B_k A_k$ | Reconstructs full weight updates via Truncated SVD to synthesize balanced $\bar{A}, \bar{B}$. |
| **FedPerfix** | Frozen Shallow Layers ($1 \dots L-4$) | MHSA Adapted | MHSA ($W_Q, W_V$) in Layers $L-3 \dots L$ | Targets deep attention projections; reduces payload by over 99.8% (under 2.36 MB/round). |
| **MedSAM Client Adapter (FCA)** | Frozen Backbone ($W_0$) | Bottleneck Residual Scaling | Local $W_{\text{down}}, W_{\text{up}}$ Aggregated | Tailors foundation encoders for volumetric CT/MRI segmentation with minimal trainable parameters. |
| **FedMedCLIP FAM** | Frozen Image Backbone + Frozen Text Anchor | Trainable Feature Adaptation Module | Aggregated FAM + Private MLP Head | Mutual KL divergence distillation prevents text-to-vision alignment collapse under label skew. |
| **FedCola Modality Bridge** | Shared Latent Space | Linear Cross-Modal Projections | Global Cross-Modal Alignment | Enables cross-institutional knowledge transfer between pathology and chest X-rays without paired data. |
| **FedDAT Dual-Adapter** | Vision-Language Bottlenecks | Cross-Attention Alignment | Aggregated Dual Adapters | Mutual cross-attention tuning aligning image representations with clinical report semantics. |

---

## 7. Drift-Resilient Federated Optimizers

Under institutional scanner heterogeneity, standard FedAvg oscillates or diverges. The following optimizers are implemented and applied directly to PEFT adapter parameters:

**FedDyn (Dynamic Objective Alignment):**

$$
\mathcal{L}_k^{\text{Dyn}}(\theta) = F_k(\theta) - \langle \nabla F_k(\bar{\theta}^t), \theta \rangle + \frac{\alpha}{2}\|\theta - \bar{\theta}^t\|_2^2
$$

$$
h^{t+1} = h^t - \alpha \frac{1}{K}\sum_{k=1}^K \left(\theta_k^{t+1} - \bar{\theta}^t\right), \quad \bar{\theta}^{t+1} = \frac{1}{|\mathcal{S}_t|}\sum_{k \in \mathcal{S}_t} \theta_k^{t+1} - \frac{1}{\alpha}h^{t+1}
$$

**FedOpt / FedAdam (Server Adaptive Momentum):**

$$
\Delta^t = \sum_{k \in \mathcal{S}_t} p_k \left(\theta_k^{t+1} - \bar{\theta}^t\right), \quad m^t = \beta_1 m^{t-1} + (1-\beta_1)\Delta^t, \quad v^t = \beta_2 v^{t-1} + (1-\beta_2)(\Delta^t)^2
$$

$$
\bar{\theta}^{t+1} = \bar{\theta}^t + \eta_g \frac{m^t}{\sqrt{v^t} + \epsilon}
$$

**SCAFFOLD (Control Variates on PEFT):** Corrects client drift using dual control variates ($c_k, c$) tracking local and global gradient trajectories:

$$
\theta_k^{t, \tau+1} = \theta_k^{t, \tau} - \eta_l (\nabla F_k(\theta_k^{t, \tau}) - c_k^t + c^t)
$$

**MOON (Model-Contrastive Federated Learning):** Regularizes representation drift by contrasting client features with the global model and prior local state:

$$
\mathcal{L}_{\text{MOON}} = \mathcal{L}_{\text{task}} - \mu \log \frac{\exp(\text{sim}(z_k^t, z_{\text{global}}^t) / \tau)}{\exp(\text{sim}(z_k^t, z_{\text{global}}^t) / \tau) + \exp(\text{sim}(z_k^t, z_k^{t-1}) / \tau)}
$$

**FedProx (Proximal Regularization):**

$$
\mathcal{L}_k^{\text{Prox}}(\theta_k) = F_k(\theta_k) + \frac{\mu}{2}\|\theta_k - \bar{\theta}^t\|_2^2
$$

---

## 8. Foundation-Anchored Prototype Metric Learning and Imbalance Losses

Exchanging raw feature representations without foundation anchoring leads to independent orthogonal rotations $Q \in O(D)$ and metric collapse. Fed-XRay uses FedProto anchored in frozen backbones with **Intra-Class Covariance-Trace Weighting**:

$$
\Sigma_{k,c} = \frac{1}{n_{k,c}} \sum_{i: y_i = c} \left(f(x_i) - p_{k,c}\right)\left(f(x_i) - p_{k,c}\right)^\top
$$

$$
\alpha_{k,c} = \frac{\frac{n_{k,c}}{\mathrm{Tr}(\Sigma_{k,c}) + \epsilon}}{\sum_{j \in \mathcal{K}_c} \frac{n_{j,c}}{\mathrm{Tr}(\Sigma_{j,c}) + \epsilon}}, \quad \bar{p}_c = \sum_{k \in \mathcal{K}_c} \alpha_{k,c} \, p_{k,c}
$$

**Dynamic Class Imbalance Loss Suite (all implemented in `src/fed_xray/core/imbalance_losses.py`):**

1. **Dynamic Adaptive Focal Loss (DAFL):** Modulates class focusing parameters dynamically as round validation recall evolves:
   $$\mathcal{L}_{\text{DAFL}} = -\alpha_c(t) (1 - p_t)^{\gamma_c(t)} \log(p_t), \quad \gamma_c(t) = \gamma_{\text{base}} + \mu (1 - \text{Recall}_c^{(t-1)})$$
2. **Bayesian Balanced Softmax (BSM):** Adjusts logits by prior institutional prevalence $\pi_y$:
   $$\mathcal{L}_{\text{BSM}} = -\log \frac{\exp(z_y + \log \pi_y)}{\sum_j \exp(z_j + \log \pi_j)}$$
3. **Class-Balanced Effective Samples Loss ($\mathcal{L}_{\text{CB}}$):** Penalizes redundancy via $(1-\beta)/(1-\beta^{n_y})$.
4. **LDAM Loss:** Applies per-class margin $C/n_y^{1/4}$ to discourage majority-class dominance.
5. **Missing-Class Repel Loss ($\mathcal{L}_{\text{repel}}$):** Enforces a minimum Euclidean separation margin $m$ from unobserved global prototypes:
   $$\mathcal{L}_{\text{repel}}(x_i, y_i; \bar{\mathcal{P}}) = \sum_{c \neq y_i} \max\left(0, m - \|f(x_i) - \bar{p}_c\|_2\right)$$

---

## 9. Privacy Architecture

Fed-XRay enforces the **Option J Architecture Standard**, combining cryptographic mechanisms with formally bounded differential privacy.

### Formally Bounded Differential Privacy

Patient-level DP clipping prevents privacy collapse across gigapixel crops or multi-study patients. The per-patient gradient is clipped before any aggregation:

$$g_i = \frac{1}{m_i} \sum_{j=1}^{m_i} g_{i, j}, \quad \bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

Gaussian noise is then added to the aggregate, calibrated to achieve $(\epsilon \le 2.0, \delta \le 10^{-5})$-DP via Rényi DP composition accounting.

> [!IMPORTANT]
> **Formally bounded, not zero-leakage privacy:** DP bounds information leakage but does not eliminate it. At $\epsilon = 2.0$, privacy leakage is constrained but non-zero. This epsilon value is chosen to minimize utility loss while maintaining a meaningful formal bound. DP always incurs a utility cost; the system is calibrated to keep this cost small at $\epsilon = 2.0$ for these datasets.

**Minimal cross-boundary data exposure:** Only PEFT adapter weights (not raw images or records) leave institutional firewalls. DP clipping bounds gradient information leakage. No raw patient data is transmitted.

### SecAgg+ ECDH Zero-Sum Masking (Simulated)

Peer-to-peer X25519 ECDH key exchange establishes shared secrets $k_{i,j}$. Using HKDF-SHA256, clients generate pairwise zero-sum pseudo-random vectors ($s_{i,j} = -s_{j,i}$) such that $\sum_{i=1}^K s_i = 0$, blinding individual updates to the server.

> [!WARNING]
> The SecAgg+ implementation in `privacy.py` runs as an **in-process Python simulation**. All clients execute within the same process; no real peer-to-peer network communication takes place. A real SecAgg+ deployment requires an authenticated multi-party network protocol with dropout handling.

### CKKS Homomorphic Encryption (Simulated via TenSEAL)

Ciphertext addition of FFA-LoRA $B$ matrix updates is performed using TenSEAL (`poly_modulus_degree=8192`, `scale_bits=40`).

> [!WARNING]
> This is a **single-key simulation**. The server generates and holds the full TenSEAL CKKS key. The server can decrypt. True multi-party threshold CKKS, where no single party holds the full decryption key, requires distributed key generation and is a planned research target.

### Enclave Attestation

The `/api/fl/enclave-handshake` endpoint verifies node public key fingerprints, HE polynomial degrees, and DP clipping bounds before training. It produces an electronic SHA-256 attestation certificate digest displayed in the clinical HUD.

---

## 10. Strategy E Datasets and Licenses

### 10.1 Dataset Sources

| Dataset | Modality | Natural Provenance / Centers | Size and Scope | License |
| :--- | :--- | :--- | :--- | :--- |
| **ISIC 2019** | Dermoscopy | BCN-20000 (Barcelona), ViDIR (Vienna), Univ. Queensland | 25,331 images, 8 diagnostic categories | CC BY-NC 4.0 (non-commercial research only) |
| **NCT-CRC-HE-100K** | Colorectal Histopathology | NCT Tissue Biobank, UMM Pathology Archive | 100,000 patches, 9 tissue classes, 86 patients | CC BY 4.0 |
| **CRC-VAL-HE-7K** | Colorectal Histopathology | Independent Holdout Center | 7,180 patches, 50 holdout patients | CC BY 4.0 |
| **MIMIC-CXR-JPG (v2.1.0)** | Chest Radiography | Beth Israel Deaconess Medical Center | Multi-view paired studies with radiology reports | PhysioNet Credentialed Health Data License 1.5.0 - requires credentialed PhysioNet access |

> [!CAUTION]
> MIMIC-CXR-JPG requires credentialed access via PhysioNet. You must complete the required training and data use agreement before accessing or using this dataset. ISIC 2019 is restricted to non-commercial research use under CC BY-NC 4.0.

**Standard Preprocessing Pipeline:** Bilinear resize to $256 \times 256$; training: random resized crop to $224 \times 224$ (scale $[0.8, 1.0]$), horizontal/vertical flips ($p=0.5$), color jitter (0.1); validation/test: center crop to $224 \times 224$; ImageNet channel standardization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$), yielding $14 \times 14 = 196$ patch tokens under patch size 16.

### 10.2 Non-IID Heterogeneity Scenarios (A to G)

| Scenario | Name | Skew Mechanism | Clinical Phenomenon |
| :--- | :--- | :--- | :--- |
| **A** | Natural Hospital Provenance | Institutional site partitioning | Preserves scanner optics, staining protocols, demographic mix. |
| **B** | Mild Label Skew | Dirichlet ($\alpha = 1.0$) | Routine multi-center outpatient variance. |
| **C** | Moderate Label Skew | Dirichlet ($\alpha = 0.3$) | Regional specialty bias (academic referral centers). |
| **D** | Severe Label Skew | Dirichlet ($\alpha = 0.05$) | Extreme single-pathology dominance (~90% local prevalence). |
| **E** | Missing Institutional Classes | Disjoint class subsets per hospital | Specialized hospitals with zero cases of certain oncology classes. |
| **F** | Global Long-Tailed Skew | Pareto distribution (100:1 ratio) | Rare cancer subtyping under national registry frequencies. |
| **G** | Combined Quantity and Extreme Skew | Variable sample sizes + severe Dirichlet | Low-resource community clinics versus high-volume oncology hospitals. |

### 10.3 Leak-Free Patient-Level Cohort Isolation

Cohorts are partitioned exclusively by `patient_id` (and `lesion_id` for dermoscopy multi-crops):

$$
\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset
$$

Split ratio: 70% train / 15% val / 15% test. Exact duplicates are rejected via SHA-256 cryptographic digests; near-duplicates via 64-bit difference hashing ($d_H \le 3$).

---

## 11. Benchmark Claims

> [!IMPORTANT]
> **All AUC and accuracy figures in the table below are literature reference targets, not measurements from this codebase's demo environment.**
>
> The demo environment uses approximately 40 samples per hospital node (160 total across 4 hospitals), which is insufficient for reliable AUC estimation. No confidence intervals are reported. These values are drawn from federated learning literature on similar datasets and architectures and are provided as aspirational research targets, not empirical results from this repository.

| Strategy | Uplink MB/Round | Communication Saving | Target AUC (Literature Ref.) |
| :--- | :--- | :--- | :--- |
| Centralized ViT-B/16 (full data) | N/A | N/A | ~0.987 (reference) |
| FedAvg Full ViT-B/16 (~85M params) | ~340 MB | 0.0% (baseline) | ~0.961 (reference) |
| Fed-XRay FFA-LoRA + FedDyn (ViT-B) | ~2.36 MB | ~99.3% | ~0.984 (reference) |
| Fed-XRay FFA-LoRA + FedDyn (ViT-Tiny) | ~0.6 MB | ~99.8% | ~0.951 (reference) |
| FedProto Centroids only | ~0.024 MB | ~99.999% | ~0.944 (reference) |

> [!WARNING]
> **Baseline fairness:** The comparison between "Fed-XRay FFA-LoRA + FedDyn" and "FedAvg Full ViT-B/16" conflates two variables: the optimization algorithm AND the number of trained parameters. A fair comparison requires equal parameter budget (same architecture, same number of trainable parameters) with only the aggregation strategy differing. The table above should be interpreted as a communication-efficiency analysis, not a controlled algorithm comparison.

**Communication Economics Reference:**

| Architecture | Uplink Payload/Client/Round | Client VRAM |
| :--- | :--- | :--- |
| Full ViT-H/14 from scratch | 2,430 MB (theoretical) | >24 GB |
| Full ViT-B/16 from scratch | ~340 MB (theoretical) | ~8 GB |
| Standard Symmetric LoRA (r=16, ViT-B) | ~4.72 MB | ~6.2 GB |
| FFA-LoRA (Frozen A, Send B, ViT-B) | ~2.36 MB | ~5.8 GB |
| FedSA-LoRA (Share-A / Local-B, ViT-B) | ~2.36 MB | ~5.8 GB |
| FedProto Centroids ($\mathbb{R}^{768}$, ViT-B) | ~0.024 MB | ~5.8 GB |

*ViT-H/14 and Swin UNETR figures are theoretical baselines. Only ViT-Tiny, ViT-Base, and CNN values have been measured in this codebase.*

---

## 12. Clinical Decision Support System (CDSS)

> [!CAUTION]
> **Research Prototype - Not for Clinical Use.** CDSS outputs (classification probabilities, saliency maps, RAG retrievals) are research artifacts. They must not inform diagnostic or treatment decisions.

### Dynamic HNSW RAG Retrieval

The `src/fed_xray/cdss/similarity.py` HNSW engine dynamically detects and indexes the full native latent dimensionality of the deployed backbone:

- ViT-Tiny: 192-dimensional latent embeddings
- ViT-Base: 768-dimensional latent embeddings
- Swin UNETR: 384-dimensional or 768-dimensional volumetric latent embeddings
- UNI Foundation Backbone: 512-dimensional or 1024-dimensional pathology embeddings
- FedMedCLIP: 512-dimensional multimodal latent embeddings
- CNN Baseline: 128-dimensional latent embeddings

Query vectors must strictly match the active index dimension; mismatches produce explicit diagnostic feedback rather than silent zero-padding. Matches are retrieved from biopsy-confirmed Strategy E reference cohorts.

### Structured Explainable AI (XAI) and Grad-CAM Hotspots

Forward and backward PyTorch tensor hooks (`register_full_backward_hook`) compute gradient-weighted class activation maps. CNN convolutional layers and ViT layer-norm multi-head self-attention projections are targeted automatically.

- **Zero-Markdown-Asterisk Policy:** Diagnostic explanations are rendered in clean, structured clinical text without raw markdown asterisks (`**`), ensuring readability across clinical HUDs and exported PDF reports.
- **Quadrant Spatial Localization:** Automatically maps relevance peak activation centers to clinical quadrants (Upper Left, Upper Right, Lower Left, Lower Right, or Central Perihilar region).
- **Multimodal Terminology Adaptation:** Differentiates diagnostic phrasing based on modality (dermoscopic atypical pigment networks / melanocytic lesions vs radiological consolidation / pleural opacities).
- **Persistent Forensic Audit Trail:** Every performed diagnosis, predicted class probabilities, Grad-CAM metrics, and clinician findings are committed to the SQLite WAL `diagnostic_audits` table and queryable via `GET /api/diagnose/audit-trail`.

> [!NOTE]
> **Grad-CAM limitations:** Grad-CAM and attention-based saliency methods are approximations. For Vision Transformers, the relationship between attention maps and model decisions is not straightforward. Saliency maps should be interpreted as model inspection tools, not as precise localizations of pathology. Clinical localization requires dedicated segmentation or detection models validated on annotated datasets. If hooks are unavailable or gradients fail to flow, the engine raises an explicit `GradCAMUnavailableError`. Synthetic Gaussian blur blobs are prohibited.

### PDF Report Engine and Voice Dictation

- **Vector PDF Reports:** Compiled via `fpdf2` with input scan embeddings, Grad-CAM overlays, argmax class probabilities, and retrieved reference cases. Reports include a SHA-256 integrity hash.
- **Bilingual Voice Dictation:** Asynchronous `gTTS` synthesis in English and Turkish with persistent disk cache and clean UI audio fallbacks.

---

## 13. Multi-Platform Interfaces

### Next.js 14 Clinical SaaS Web App (`frontend/`)

Built with React 18, TypeScript, Tailwind CSS, and Next.js 14 App Router:

- **Landing Page (`/`):** Consortium topology, security governance overview, 5 verifiable KPI metrics, interactive results showcase tabs, and backend connection verification.
- **Consortium Launch Enclave Modal:** Executes 4-stage cryptographic enclave handshakes via `/api/fl/enclave-handshake`, rendering ECDH fingerprints and SHA-256 attestation digests.
- **Telemetry Dashboard (`/dashboard`):** Real-time SSE streaming and polling HUD displaying round loss, accuracy, client drift, and Byzantine threat status.
- **Diagnostic Studio:** Upload scans, view Grad-CAM heatmaps, inspect RAG digital twins, listen to audio briefings, and download PDF reports.
- **Browser Runtime Shield (`runtimeShield.ts`):** Intercepts noisy third-party Chrome/Edge browser extension errors (e.g. `useCache`, `content.js`), isolates service worker port closures (`message channel closed`), and deduplicates console warnings.
- **React Subtree Resilience & Error Boundaries:** Route segment error containment via `app/error.tsx` and `app/global-error.tsx`, paired with `ClinicalErrorBoundary` encapsulating critical dashboard stages and landing tabs with bilingual recovery controls. All array operations implement defensive guards against nullable payloads.
- **Universal Responsiveness:** Tested across Mobile (<768px), Tablet (768-1024px), Laptop (1024-1440px), and Desktop/4K (>=1440px) viewports without horizontal clipping.

### Streamlit Scientific Benchmark UI (`app.py`)

A standalone research workbench for multi-algorithm comparative evaluation:

- Native Strategy E clinical data loader with patient-level isolation.
- Interactive optimizer comparative benchmarking (FedAvg, FedProx, SCAFFOLD, FedDyn, FedAdam, MOON).
- Visualizer for 7 Non-IID Skew Scenarios with class distribution charts.

---

## 14. Flower Protocol Federation Bridge

Complete bidirectional interoperability with the official Flower (`flwr`) framework via `src/fed_xray/federation/flower_protocol_bridge.py`:

```
[ Flower Aggregator (Server) ]
              |
+-------------+-------------+
|             |             |
[ InProcessProxy ]  [ gRPC Proxy ]  [ gRPC Proxy ]
|             |             |
[ FlowerHospitalClient ] [ Remote Client ] [ Remote Client ]
|
[ HospitalClient (PEFT/DP/HE) ]
```

- **FlowerHospitalClient:** Implements `flwr.client.NumPyClient`, serializing PyTorch parameter dictionaries to/from NumPy ndarrays.
- **InProcessClientProxy:** High-speed zero-IPC multi-client execution within a single Python process.
- **StrategyEHospitalStrategy:** Custom aggregation strategy integrating patient-level DP and Byzantine filtering.

```bash
# Run multi-hospital federation via Flower CLI
python scripts/run_flower_federation.py --rounds 5 --hospitals 4 --model vit_tiny
```

FastAPI endpoints: `POST /api/fl/flower-start` and `GET /api/fl/flower-info`.

---

## 15. REST and Streaming API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root API information, service status, and documentation pointers. |
| `GET` | `/api/health` | System health, PyTorch execution device, and CUDA status. |
| `GET` | `/api/state` | Active federation state, node counts, persistence status, and model parameters. |
| `GET` | `/api/state/history` | Full persistent round telemetry history retrieved from SQLite WAL storage. |
| `POST` | `/api/state/reset` | Resets persistent SQLite state store and clears active consortium session. |
| `POST` | `/api/state/save` | Explicitly persists trained weights, metadata, cohorts, and history to disk/SQLite. |
| `POST` | `/api/state/restore` | Restores trained weights, metadata, cohorts, and history from disk/SQLite. |
| `POST` | `/api/cohorts/generate` | Partitions Strategy E clinical cohorts across Scenarios A through G. |
| `GET` | `/api/cohorts/verify-authenticity` | Forensic audit of staged cohort provenance, patient isolation, and authenticity. |
| `POST` | `/api/cohorts/stage-real-isic` | Stages authentic multi-center ISIC 2019 dermoscopy cohorts. |
| `GET` | `/api/fl/enclave-handshake` | Cryptographic enclave attestation: X25519 fingerprints, SHA-256 certificate digest. |
| `POST` | `/api/fl/train-start` | Launches asynchronous federated optimization background worker. |
| `GET` | `/api/fl/train-status` | Polls training progress, round telemetry, and Byzantine threat reports. |
| `GET` | `/api/fl/train-stream` | Server-Sent Events (SSE) stream: real-time round metrics, loss, and client drift. |
| `GET` | `/api/fl/privacy-status` | Differential Privacy bounds, Rényi DP accounting budget, and CKKS parameters. |
| `POST` | `/api/fl/flower-start` | Initiates multi-hospital federation via official Flower framework bridge. |
| `GET` | `/api/fl/flower-info` | Reports Flower framework bridge status, strategies, and runtime metrics. |
| `POST` | `/api/cdss/diagnose` | Forward pass classification, structured XAI explanation, and Grad-CAM heatmap. |
| `GET` | `/api/diagnose/audit-trail` | Forensic audit trail of historical clinical diagnoses committed to SQLite. |
| `GET` | `/api/cdss/rag-similar` | Retrieves top-k biopsy-confirmed digital twins from HNSW vector index. |
| `GET` | `/api/cdss/voice` | Streams synthesized bilingual diagnostic audio briefing (EN/TR). |
| `GET` | `/api/cdss/report-pdf` | Compiles and streams cryptographic vector PDF diagnostic report with SHA-256 hash. |

All request/response schemas are defined in `api/schemas.py` as Pydantic v2 models.

---

## 16. Test Suite Matrix (326 Tests)

Fed-XRay maintains 326 automated tests (246 Python Pytest backend + 80 TypeScript Vitest frontend), all passing at 100%.

### Backend Pytest Suites (36 Test Files, 246 Tests)

| Subsystem | Test File | Count | Scope |
| :--- | :--- | :--- | :--- |
| Baseline and Models | `test_all.py` | 7 | Baseline model creation, Grad-CAM hooks, aggregation consistency, and Byzantine defense. |
| API Exhaustive | `test_api_deep_suite.py` | 27 | Parameter permutations, SSE bounds, error recovery, path traversal blocks, and schema bounds. |
| API Core Routing | `test_api_endpoints.py` | 5 | REST routing, health, cohort partitioning, diagnostic responses, and PDF streaming. |
| Pydantic Contracts | `test_api_schemas_contracts.py` | 5 | Pydantic v2 validation, boundary conditions, serialization integrity, and opacity clamping. |
| Asymmetric PEFT | `test_asymmetric_foundation_peft.py` | 10 | FlexLoRA SVD, FedAS-LoRA RSS routing, HetLoRA rank slicing, FedPerfix, and foundation integration. |
| Benchmark Engine | `test_benchmarking_engine.py` | 1 | Foundation PEFT empirical benchmark execution and metric logging. |
| Browser Runtime Shield | `test_browser_runtime_shield_and_extension_isolation.py` | 4 | Interception of extension errors, port closures, console deduplication, and boundary contracts. |
| CDSS Explainability | `test_cdss_explainability_and_localization.py` | 6 | Zero-markdown-asterisk invariant, structured clinical formatting, quadrant spatial localization. |
| PDF Reporting | `test_cdss_report.py` | 11 | PDF compilation, image overlays, SHA-256 digests, empty state handling, and vector rendering. |
| RAG and Similarity | `test_cdss_similarity.py` | 12 | Dynamic HNSW indexing (128 to 1024 dims), biopsy twin search, cosine distance verification. |
| Voice Dictation | `test_cdss_voice.py` | 4 | Async speech synthesis, disk caching, and bilingual audio fallbacks. |
| Explainable AI | `test_cdss_xai.py` | 3 | Multi-layer Grad-CAM, forward/backward hooks, and explicit exception raising. |
| Client Subtree Resilience | `test_client_subtree_resilience_and_error_boundaries.py` | 4 | Next.js app error boundaries, granular component containment, defensive array access. |
| Clinical Cohort Staging | `test_clinical_cohort_authenticity_and_staging.py` | 7 | Real-world dataset staging, forensic cohort authenticity auditing, honest non-clinical tagging. |
| Consortium Cryptography | `test_consortium_cryptography_and_zero_plaintext.py` | 3 | RLWE 128-bit CKKS encryption, SecAgg+ zero-sum masking, Byzantine defense under ciphertext. |
| Persistent State Store | `test_consortium_persistent_state_store.py` | 9 | SQLite WAL persistence, relational tables, restart state hydration, history and reset endpoints. |
| Imbalance Losses | `test_core_imbalance_losses.py` | 6 | DAFL, BSM, LDAM, and missing-class repel loss numerical stability. |
| Cryptographic Privacy | `test_crypto_privacy_optimizers.py` | 11 | TenSEAL CKKS ciphertext addition, SecAgg+ zero-sum masking, distributed DP Gaussian noise. |
| E2E Lifecycle | `test_e2e_full_federated_pipeline.py` | 2 | End-to-end ingestion, federated training, RAG digital twins, and PDF report compilation. |
| Enclave Attestation | `test_enclave_handshake.py` | 1 | Enclave handshake, X25519 fingerprint verification, SHA-256 certificates. |
| CDSS Grounding | `test_evidence_grounded_cdss.py` | 3 | Grad-CAM spatial peak relevance, genuine RAG case retrieval, and honest failure modes. |
| Flower Protocol | `test_flower_and_sota_enhancements.py` | 7 | Flower bridge, InProcessProxy, NumPy parameter serialization roundtrips. |
| Checkpoint Integrity | `test_foundation_checkpoint_integrity_and_provenance.py` | 10 | Zero-silent-pass policy, honest exception raising on missing weights, SHA-256 weight fingerprinting. |
| Foundation Privacy | `test_foundation_privacy.py` | 16 | Multimodal semantic anchoring, CKKS vector scaling, and Rényi Differential Privacy composition bounds. |
| Strategy E Generator | `test_generator_strategy_e.py` | 12 | 224x224 input resolution, 196 ViT patch tokens, robust PyTorch DataLoader pipelines. |
| Multimodal PEFT Adapters | `test_multimodal_foundation_peft_adapters.py` | 5 | Runtime integration of MedSAM, FedMedCLIP, FedCola, and FedDAT dual-attention adapters. |
| Network Routing | `test_network_routing_and_tunnel_resilience.py` | 6 | Dynamic ephemeral tunnel detection, dead domain blacklisting, automatic loopback fallback, open CORS. |
| Numerical Stability | `test_numerical_stability_losses.py` | 6 | Boundary conditions, extreme class imbalance limits, and high-dimensional vector stability. |
| Prototypes and Losses | `test_prototypes.py` | 7 | FedProto dispersion weighting, covariance traces, centroid synthesis, and prototype rounds. |
| Real-World Scenarios | `test_real_world_data_scenarios.py` | 8 | DiskMedicalDataset manifests, Non-IID Scenarios A through G, SHA-256 image deduplication. |
| Robustness and Adversarial | `test_robustness_and_adversarial.py` | 5 | Byzantine gradient poisoning mitigation, NaN/Inf gradient resilience, free-rider rejection. |
| State Persistence | `test_state_persistence.py` | 4 | AppState save and restore roundtrips, persistence endpoint verification. |
| Strategy E Ingestion | `test_strategy_e_real_ingestion.py` | 7 | ISIC Archive API v2 ingestion, file staging, and leak-free patient isolation invariants. |
| Swin UNETR Backbone | `test_swin_unetr_radiology_backbone.py` | 4 | Authentic hierarchical Shifted Window Attention, patch merging, 2D/3D volumetric support, and PEFT adaptation. |
| ViT PEFT Optimizers | `test_vit_peft_optimizers.py` | 4 | ViT-Base FFA-LoRA zero bilinear error, FedDyn, SCAFFOLD, and MOON optimization. |
| ViT-Tiny FFA-LoRA | `test_vit_tiny_ffa_lora.py` | 4 | ViT-Tiny 224x224 FFA-LoRA injection, CPU runtime efficiency, state dict roundtrips. |

### Frontend Vitest Suites (19 Test Files, 80 Tests)

| Component / Utility | Test File | Count | Scope |
| :--- | :--- | :--- | :--- |
| API Client | `api.test.ts` | 6 | Dual SSE streaming, polling fallbacks, honest error parsing, and timeout resilience. |
| Clinical Error Boundary | `ClinicalErrorBoundary.test.tsx` | 3 | Catches rendering errors, renders bilingual recovery controls, reset callbacks. |
| Enclave Overlay | `ConsortiumLaunchOverlay.test.tsx` | 5 | 4-stage cryptographic enclave handshake, ECDH fingerprints, attestation badges. |
| Diagnostic Studio | `DiagnosticStudio.test.tsx` | 5 | Pre-training locked state, forward pass inference, Grad-CAM overlays, audio briefing, PDF trigger. |
| Hero Component | `Hero.test.tsx` | 3 | Headline rendering, consortium status badge, quick launch actions. |
| Hospital Studio | `HospitalStudio.test.tsx` | 3 | Multi-hospital tabs, demographic breakdown, Non-IID distribution charts. |
| Presentation Page | `LandingPage.test.tsx` | 5 | Verifiable KPI metrics, interactive results showcase tabs, enclave launch trigger. |
| Live HUD | `LiveHud.test.tsx` | 3 | Attestation digest badge, real-time telemetry indicators, pulse status. |
| Geometry Utilities | `lungGeometry.test.ts` | 6 | Anatomical lung SVG boundary calculations and coordinate scaling. |
| 3D Volumetric Lung Scene | `LungViewer3D.test.tsx` | 3 | Three.js 3D volumetric lung container rendering and cleanup. |
| Navigation | `Navbar.test.tsx` | 6 | Bilingual language toggle (EN/TR), dark/light mode toggle, responsive mobile navigation. |
| RAG Digital Twins | `RagDigitalTwins.test.tsx` | 2 | Biopsy-confirmed digital twin rendering, confidence bars, clinical metadata tags. |
| Report Download | `ReportDownload.test.tsx` | 3 | PDF report card rendering, encoded download link generation, disabled states. |
| Browser Runtime Shield | `runtimeShield.test.ts` | 9 | Interception of extension errors, port drop isolation, console deduplication. |
| Subtree Isolation | `subtreeIsolation.test.tsx` | 5 | Fault containment across null/undefined sample images, twin lists, and telemetry arrays. |
| Telemetry Cockpit | `TelemetryCockpit.test.tsx` | 3 | Loss and accuracy convergence graphs, client drift tracking, Byzantine threat alerts. |
| Localization | `translations.test.ts` | 4 | Complete key parity and translation completeness across English and Turkish dictionaries. |
| Voice Assistant | `VoiceAssistant.test.tsx` | 2 | Audio player controls, bilingual voice toggle, synthesization loading indicators. |
| Workflow Stepper | `WorkflowStepper.test.tsx` | 4 | 4-step workflow navigation, step completion locking, interactive advancement. |

---

## 17. Repository Directory Structure

```text
Fed-XRay/
|-- api/                               # FastAPI Async Backend
|   |-- main.py                        # REST, SSE, Enclave, Persistence, and CDSS Routes
|   `-- schemas.py                     # Pydantic v2 Request/Response Contracts
|-- frontend/                          # Next.js 14 Clinical SaaS Web App
|   |-- src/
|   |   |-- app/                       # App Router Pages, Error Boundaries (error.tsx, global-error.tsx)
|   |   |-- components/                # ClinicalErrorBoundary, Cockpit Panels, SVG Topology, HUD
|   |   |-- context/                   # Language (EN/TR) and API Connection Providers
|   |   |-- lib/                       # API Client, Runtime Shield, Translations, Utilities
|   |   |-- types/                     # TypeScript Strict Interface Definitions
|   |   `-- __tests__/                 # Vitest Component and Integration Tests (80 Tests)
|   |-- package.json
|   `-- tailwind.config.js             # Clinical SaaS HSL Design Token System
|-- scripts/
|   `-- run_flower_federation.py       # Flower Multi-Hospital Federation Runner
|-- src/
|   `-- fed_xray/                      # Core Python Package
|       |-- core/                      # Federated Optimization Core
|       |   |-- client.py              # HospitalClient (PEFT, FedDyn, SCAFFOLD, MOON, DP, MedSAM, FedMedCLIP)
|       |   |-- server.py              # CentralServer (FedAdam, FedDyn, Byzantine Detection, Asymmetric PEFT)
|       |   |-- privacy.py             # Option J: TenSEAL CKKS HE, SecAgg+ ECDH, Rényi DP
|       |   |-- prototypes.py          # FedProto Dispersion-Weighted Synthesis
|       |   |-- imbalance_losses.py    # DAFL, Balanced Softmax, LDAM, Missing-Class Repel
|       |   `-- metrics.py             # Telemetry Trackers and Anomaly Reports
|       |-- storage/
|       |   `-- state_store.py         # ACID PersistentStateStore (SQLite WAL Mode across 5 Relational Tables)
|       |-- federation/
|       |   `-- flower_protocol_bridge.py  # Flower (flwr) Client Proxy and Strategy
|       |-- models/
|       |   |-- vit.py                 # Vision Transformer (MHSA, PatchEmbedding, create_medical_vit)
|       |   |-- peft.py                # FFA-LoRA, FedSA-LoRA, FedAS-LoRA, FlexLoRA, FedPerfix, MedSAM FCA
|       |   |-- foundation.py          # Swin UNETR 3D, UNI Backbone, FedMedCLIP, FedCola, FedDAT
|       |   |-- precomputed_clinical_anchors.pt # Verified Bio_ClinicalBERT Semantic Embeddings
|       |   `-- cnn.py                 # XRayClassifier Baseline
|       |-- data/
|       |   |-- real_world.py          # DiskMedicalDataset, Scenarios A-G, SHA-256 Deduplication
|       |   |-- ingest_real_world.py   # Strategy E Dataset Staging and In-Silico Calibration Suite
|       |   `-- generator.py           # Multi-Resolution 224x224 ViT-Compatible Generator
|       `-- cdss/
|           |-- xai.py                 # Grad-CAM Heatmaps and Activation Overlays (Structured, Zero-Asterisks)
|           |-- similarity.py          # Dynamic HNSW Digital Twin Retrieval (128 to 1024 dims)
|           |-- report.py              # Cryptographic Vector PDF Report Engine (fpdf2 + SHA-256 Digest)
|           |-- voice.py               # Speech Dictation Engine (Async gTTS with Disk Cache)
|           `-- i18n.py                # Bilingual Clinical Localization (EN/TR)
|-- benchmarks/
|   `-- benchmark_foundation_peft.py   # Foundation PEFT Comparative Benchmark Suite
|-- tests/                             # Pytest Test Suites (36 Files, 246 Tests)
|-- research/                          # Canonical Scientific Research Foundations (6 Papers)
|   |-- federated_vision_transformers_for_oncology.md
|   |-- federated_foundation_models_for_cancer.md
|   |-- privacy_preserving_federated_learning_with_dp_he_peft_for_cancer_imaging.md
|   |-- personalized_prototype_federated_learning_for_imbalanced_medical_ai.md
|   |-- federated_medical_vision_language_models_and_retrieval_augmented_oncology_ai.md
|   `-- real_world_medical_dataset_migration_for_federated_cancer_ai.md
|-- app.py                             # Standalone Streamlit Scientific Benchmark UI
|-- render.yaml                        # Cloud Deployment Blueprint
|-- Dockerfile                         # Container Deployment Configuration
`-- requirements.txt                   # Production Python Dependencies
```

---

## 18. Installation and Quick Start

### 18.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/yusufcalisir/Fed-XRay.git
cd Fed-XRay

# Create and activate isolated Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install production Python dependencies
pip install -r requirements.txt
```

### 18.2 Automated Test Execution

```bash
# Execute all 246 Python backend tests
python -m pytest tests/ -v

# Execute all 80 Next.js frontend tests
cd frontend
npm install
npm test -- --run
cd ..
```

### 18.3 Flower Protocol Federation

```bash
python scripts/run_flower_federation.py --rounds 5 --hospitals 4 --model vit_tiny
```

### 18.4 Foundation PEFT Benchmark

```bash
python -m benchmarks.benchmark_foundation_peft
```

### 18.5 Launching Services

**Option A: Next.js 14 Clinical SaaS Web App (Recommended)**

```bash
# Terminal 1 - FastAPI Backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 - Next.js Frontend
cd frontend
npm run dev
# Open http://localhost:3000
```

**Option B: Standalone Streamlit Benchmark UI**

```bash
streamlit run app.py
# Open http://localhost:8501
```

**Option C: Docker Container**

```bash
docker build -t fed-xray:latest .
docker run -p 8000:8000 -p 3000:3000 fed-xray:latest
```

---

## 19. Scientific Anchors and Citations

If you use Fed-XRay in your research, please cite the foundational papers:

1. **Hu et al.** (ICLR 2022) - *LoRA: Low-Rank Adaptation of Large Language Models.*
2. **Sun et al.** (ICLR 2024) - *FFA-LoRA: Freezing Factor A in Federated Low-Rank Adaptation.*
3. **Guo et al.** (ICLR 2025) - *FedSA-LoRA: Factor-Wise Asymmetric Federated Fine-Tuning.*
4. **Sun et al.** (ICCV 2023) - *FedPerfix: Towards Partial Model Personalization for Vision Transformers.*
5. **Chen et al.** (Nature Medicine 2024) - *UNI: General-purpose Self-Supervised Vision Transformer for Pathology.*
6. **Vorontsov et al.** (Nature Medicine 2024) - *Virchow: A Pan-Cancer Pathology Foundation Model.*
7. **Lu et al.** (Nature Medicine 2024) - *CONCH: Visual Language Foundation Model for Histopathology.*
8. **Tan et al.** (AAAI 2022) - *FedProto: Federated Prototype Learning across Heterogeneous Clients.*
9. **Acar et al.** (ICLR 2021) - *Federated Learning Based on Dynamic Regularization (FedDyn).*
10. **Li et al.** (CVPR 2021) - *Model-Contrastive Federated Learning (MOON).*
11. **Mironov** (CSF 2017) - *Rényi Differential Privacy.*
12. **McMahan et al.** (AISTATS 2017) - *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).*
13. **Li et al.** (ICLR 2020) - *Federated Optimization in Heterogeneous Networks (FedProx).*
14. **Karimireddy et al.** (ICML 2020) - *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.*
15. **Reddi et al.** (ICLR 2021) - *Adaptive Federated Optimization (FedOpt/FedAdam).*
16. **Dong et al.** (ICLR 2023) - *FedALA: Federated Adaptive Local Aggregation.*
17. **Bonawitz et al.** (CCS 2017) - *Practical Secure Aggregation for Privacy-Preserving Machine Learning (SecAgg).*
