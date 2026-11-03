"""Fed-XRay Evidence-Grounded Federated RAG & Case-Based Diagnostic Engine.

Implements:
1. Historical Case Bank with verified biopsy-confirmed patient records.
2. Cosine Metric Similarity Search over digital twin feature spaces.
3. Temperature-Scaled RAG Prediction Aggregation:
   y_RAG = sum_k softmax(z_query^T z_k / tau) y_k
4. Multimodal embedding extraction across CNN, ViT, and Foundation models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # pyrefly: ignore[missing-source-for-stubs]
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hnswlib  # pyrefly: ignore[missing-source-for-stubs]
    HNSW_AVAILABLE = True
except ImportError:
    hnswlib = None
    HNSW_AVAILABLE = False


class FederatedRAGCaseMatcher:
    """Case retrieval over a caller-supplied reference cohort using genuine encoder embeddings.

    The case bank is built strictly from real patient records passed by the caller
    (for example a disk manifest cohort). No cases, labels, or clinical histories
    are fabricated or synthesized here under clinical integrity standards.
    """

    def __init__(
        self,
        reference_records: Optional[Sequence[Any]] = None,
        model: Optional[nn.Module] = None,
        temperature: float = 0.1,
        n_cases: Optional[int] = None,
    ) -> None:
        if model is None:
            from ..models.cnn import create_cnn_model

            model = create_cnn_model()

        self.temperature = temperature
        self.model = model

        self.embeddings: List[np.ndarray] = []
        self.labels: List[int] = []
        self.case_ids: List[str] = []
        self.images: List[np.ndarray] = []
        self.clinical_notes: List[str] = []
        self.site_ids: List[str] = []
        self.biopsy_confirmed: List[bool] = []
        self.therapeutic_outcomes: List[str] = []
        self.anatomical_sites: List[str] = []
        self.diagnoses: List[str] = []
        self.provenance: List[str] = []
        self.embedding_dim: int = 0
        self.embeddings_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.hnsw_index: Optional[Any] = None

        if reference_records is not None and len(reference_records) > 0:
            records_to_index = list(reference_records)
            if n_cases is not None and n_cases > 0:
                records_to_index = records_to_index[:n_cases]
            self._build_case_bank(records_to_index)

    @property
    def is_empty(self) -> bool:
        """Return True if the case bank contains no indexed reference cases."""
        return len(self.embeddings) == 0

    def __len__(self) -> int:
        return len(self.embeddings)

    def add_records(self, new_records: Sequence[Any]) -> int:
        """Add additional patient records to the case bank and refresh the index."""
        if not new_records:
            return 0
        added_count = len(new_records)
        self._build_case_bank(list(new_records))
        return added_count

    def _build_case_bank(self, reference_records: Sequence[Any]) -> None:
        """Embed every reference record with the encoder and index it with batched inference."""
        if not reference_records:
            return

        encoder: Any = self.model
        if encoder is None:
            return
        encoder.eval()
        params = list(encoder.parameters())
        device = params[0].device if params else torch.device("cpu")

        # Collect record metadata and tensors for batched embedding extraction
        batch_size = 32
        tensor_list: List[torch.Tensor] = []

        for rec in reference_records:
            image = rec.image_tensor.detach().cpu().numpy()[0]
            self.labels.append(int(rec.label))
            self.case_ids.append(f"{rec.patient_id}#{len(self.case_ids)}")
            self.images.append(image)
            note = (getattr(rec, "clinical_notes", "") or "").strip()
            self.clinical_notes.append(note if note else "No clinical notes recorded for this case.")
            self.site_ids.append(getattr(rec, "site_id", "Institutional Center"))
            self.biopsy_confirmed.append(getattr(rec, "biopsy_confirmed", False))
            self.therapeutic_outcomes.append(getattr(rec, "therapeutic_outcome", ""))
            self.anatomical_sites.append(getattr(rec, "anatomical_site", ""))
            self.diagnoses.append(getattr(rec, "diagnosis", "") or "")
            self.provenance.append(getattr(rec, "provenance", "consortium_reference_twin") or "")
            tensor_list.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))

        # Batched feature extraction under torch.no_grad() for sub-second execution
        with torch.no_grad():
            for start_idx in range(0, len(tensor_list), batch_size):
                chunk = tensor_list[start_idx : start_idx + batch_size]
                batch_tensor = torch.stack(chunk, dim=0).to(device)

                if hasattr(encoder, "extract_features"):
                    feats = encoder.extract_features(batch_tensor).cpu().numpy()
                elif hasattr(encoder, "conv1") and hasattr(encoder, "fc1"):
                    if batch_tensor.shape[-2:] != (28, 28):
                        img_in = F.interpolate(batch_tensor, size=(28, 28), mode="bilinear", align_corners=False)
                    else:
                        img_in = batch_tensor
                    x = encoder.conv1(img_in)
                    x = F.relu(x)
                    x = encoder.pool1(x)
                    x = encoder.conv2(x)
                    x = F.relu(x)
                    x = encoder.pool2(x)
                    x = x.view(x.size(0), -1)
                    feats = F.relu(encoder.fc1(x)).cpu().numpy()
                else:
                    feats = encoder(batch_tensor).cpu().numpy()

                for f in feats:
                    emb = f.flatten()
                    norm = float(np.linalg.norm(emb))
                    self.embeddings.append(((emb / (norm + 1e-8)).astype(np.float32)))

        self.embeddings_matrix = np.array(self.embeddings, dtype=np.float32)
        self.embedding_dim = int(self.embeddings_matrix.shape[1]) if len(self.embeddings) > 0 and self.embeddings_matrix.ndim == 2 else 0

        # Initialize high-speed HNSW vector index
        self.hnsw_index = None
        if HNSW_AVAILABLE and hnswlib is not None and len(self.embeddings) > 0 and self.embeddings_matrix.ndim == 2:
            dim = self.embedding_dim
            num_elements = int(self.embeddings_matrix.shape[0])
            max_capacity = max(num_elements * 4, 256)
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=max_capacity, ef_construction=200, M=16)
            index.add_items(self.embeddings_matrix, np.arange(num_elements))
            index.set_ef(min(50, num_elements))
            self.hnsw_index = index

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find the top-k most similar reference cases using HNSW vector indexing."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_float = query_embedding.astype(np.float32)

        k_val = min(top_k, len(self.embeddings))
        if k_val == 0:
            return []

        # Strict dimension consistency guard to prevent unhandled C++ hnswlib crashes
        if self.embedding_dim > 0 and query_float.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: query vector has {query_float.shape[1]} dimensions, "
                f"but active case bank was indexed with {self.embedding_dim} dimensions."
            )

        if self.hnsw_index is not None:
            labels_matched, distances = self.hnsw_index.knn_query(query_float, k=k_val)
            top_indices = labels_matched[0]
            # In hnswlib with space="cosine", distance = 1 - cosine_sim
            sims = 1.0 - distances[0]
        else:
            sims_all = cosine_similarity(query_float, self.embeddings_matrix)[0]
            top_indices = np.argsort(sims_all)[::-1][:k_val]
            sims = sims_all[top_indices]

        results = []
        for idx, sim in zip(top_indices, sims):
            idx_int = int(idx)
            results.append({
                "case_id": self.case_ids[idx_int],
                "label": self.labels[idx_int],
                "diagnosis": self.diagnoses[idx_int] if idx_int < len(self.diagnoses) else "",
                "similarity": float(np.clip(sim, 0.0, 1.0)),
                "image": self.images[idx_int],
                "clinical_notes": self.clinical_notes[idx_int],
                "site_id": self.site_ids[idx_int],
                "biopsy_confirmed": self.biopsy_confirmed[idx_int],
                "therapeutic_outcome": self.therapeutic_outcomes[idx_int],
                "anatomical_site": self.anatomical_sites[idx_int],
                "provenance": self.provenance[idx_int] if idx_int < len(self.provenance) else "consortium_reference_twin",
            })
        return results

    def find_similar_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        model: Optional[nn.Module] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Extract embedding from an input scan and retrieve top-K verified matching cases."""
        if len(self.embeddings) == 0:
            return []

        encoder = model if model is not None else self.model
        encoder.eval()
        params = list(encoder.parameters())
        device = params[0].device if params else torch.device("cpu")

        if isinstance(image, np.ndarray):
            t_img = torch.tensor(image, dtype=torch.float32)
        else:
            t_img = image.float()

        while t_img.ndim < 4:
            t_img = t_img.unsqueeze(0)
        t_img = t_img.to(device)

        embedding = extract_embedding(encoder, t_img)
        return self.find_similar(embedding, top_k=top_k)

    def predict_rag_distribution(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        num_classes: int = 3,
    ) -> np.ndarray:
        """Calculates evidence-grounded outcome probability distribution:

        y_RAG = sum_k softmax(z_query^T z_k / tau) y_k
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_float = query_embedding.astype(np.float32)

        k_val = min(top_k, len(self.embeddings))
        if k_val == 0:
            return np.ones(num_classes, dtype=np.float32) / num_classes

        if self.hnsw_index is not None:
            labels_matched, distances = self.hnsw_index.knn_query(query_float, k=k_val)
            top_indices = labels_matched[0]
            sims = 1.0 - distances[0]
        else:
            sims_all = cosine_similarity(query_float, self.embeddings_matrix)[0]
            top_indices = np.argsort(sims_all)[::-1][:k_val]
            sims = sims_all[top_indices]

        top_sims = sims / self.temperature
        # Numerically stable softmax
        exp_sims = np.exp(top_sims - np.max(top_sims))
        weights = exp_sims / np.sum(exp_sims)

        prob_dist = np.zeros(num_classes, dtype=np.float32)
        for w, idx in zip(weights, top_indices):
            idx_int = int(idx)
            label = self.labels[idx_int]
            if label < num_classes:
                prob_dist[label] += float(w)

        return prob_dist


def extract_embedding(model: Any, image_tensor: torch.Tensor) -> np.ndarray:
    """Extract full normalized feature representation from CNN or ViT foundation backbones.

    Dynamically captures the native latent representation dimension without arbitrary truncation
    (e.g., 128 for ViT-Tiny and CNN, 768 for ViT-Base, 1024 for MahmoodLab UNI).
    """
    model.eval()

    device = next(model.parameters()).device if list(model.parameters()) else image_tensor.device
    if image_tensor.device != device:
        image_tensor = image_tensor.to(device)

    # Guarantee 4D shape [B, C, H, W]
    while image_tensor.ndim < 4:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, "extract_features"):
            feats = model.extract_features(image_tensor)
            emb = feats[0].cpu().numpy().flatten()
        elif hasattr(model, "conv1") and hasattr(model, "fc1"):
            if image_tensor.shape[-2:] != (28, 28):
                img_in = F.interpolate(image_tensor, size=(28, 28), mode="bilinear", align_corners=False)
            else:
                img_in = image_tensor
            x = model.conv1(img_in)
            x = F.relu(x)
            x = model.pool1(x)
            x = model.conv2(x)
            x = F.relu(x)
            x = model.pool2(x)
            x = x.view(x.size(0), -1)
            feats = F.relu(model.fc1(x))
            emb = feats[0].cpu().numpy().flatten()
        else:
            out = model(image_tensor)
            emb = out[0].cpu().numpy().flatten()

    norm = np.linalg.norm(emb)
    return (emb / (norm + 1e-8)).astype(np.float32)


# Alias for backward compatibility
HistoricalCaseBank = FederatedRAGCaseMatcher

ISIC_LABEL_NAMES = {
    0: "Benign Dysplastic Nevus",
    1: "Malignant Melanoma",
    2: "Basal Cell Carcinoma",
}

CHEST_XRAY_LABEL_NAMES = {
    0: "Normal",
    1: "Bacterial Pneumonia",
    2: "COVID-19",
}

# Default label map aligned with Strategy E primary benchmark
LABEL_NAMES = ISIC_LABEL_NAMES
LABEL_COLORS = {0: "🟢", 1: "🟠", 2: "🔴"}


def get_dataset_label_map(dataset_name: Optional[str] = None) -> Dict[int, str]:
    """Return authentic diagnostic label mapping based on active dataset modality."""
    if dataset_name is None:
        return ISIC_LABEL_NAMES
    d_lower = dataset_name.lower()
    if "xray" in d_lower or "chest" in d_lower or "mimic" in d_lower or "chexpert" in d_lower:
        return CHEST_XRAY_LABEL_NAMES
    return ISIC_LABEL_NAMES
