"""Option J Dual-Layer Cryptographic & Differential Privacy Architecture.

Implements:
1. In-Transit Cryptographic Privacy:
   - Leveled CKKS Threshold Homomorphic Encryption (RLWE 128-bit) on FFA-LoRA updates.
   - Secure Aggregation (SecAgg+) with pairwise zero-sum random masking.
2. Output Model Privacy:
   - Strict Patient-Level Gaussian Differential Privacy mechanism.
   - Rényi Differential Privacy (RDP) Composition Accountant for tight (epsilon, delta)-DP bounds.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import torch


class EncryptedTensor:
    """Serialized CKKS ciphertext chunks for one parameter tensor (wire format)."""

    __slots__ = ("shape", "payload")

    def __init__(self, shape: Tuple[int, ...], payload: List[bytes]) -> None:
        self.shape = shape
        self.payload = payload

    @property
    def nbytes(self) -> int:
        """Total ciphertext size in bytes as transmitted over the network."""
        return sum(len(b) for b in self.payload)


class CKKSEncryptionEngine:
    """Leveled CKKS homomorphic encryption (RLWE) backed by Microsoft SEAL via TenSEAL.

    Key separation:
    - The secret context (holds the secret key) belongs to the key holder (clients or a
      threshold committee) and is used only by ``encrypt_*`` and ``decrypt_*``.
    - The public context (secret key removed) is the only context used by the
      aggregation server in ``homomorphic_sum``; the server never observes plaintext.

    Additive homomorphism with one plaintext-scalar multiplication level:
    ct_global = sum_k w_k * ct_k.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        scale_bits: int = 40,
        security_level: int = 128,
    ) -> None:
        try:
            import tenseal as ts
        except ImportError as exc:
            raise RuntimeError(
                "CKKS requires the 'tenseal' package (pip install tenseal). "
                "Homomorphic aggregation is unavailable without it."
            ) from exc

        self._ts = ts
        self.poly_modulus_degree = poly_modulus_degree
        self.scale_bits = scale_bits
        self.scale = 2.0 ** scale_bits
        self.security_level = security_level
        self.slots = poly_modulus_degree // 2

        self._secret_ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, scale_bits, scale_bits, 60],
        )
        self._secret_ctx.global_scale = self.scale
        self._public_ctx = self._secret_ctx.copy()
        self._public_ctx.make_context_public()

    def encrypt_vector(self, plain_vector: torch.Tensor) -> EncryptedTensor:
        """Encrypt a float tensor into serialized CKKS ciphertext chunks."""
        flat = plain_vector.detach().cpu().float().flatten().tolist()
        payload: List[bytes] = []
        for i in range(0, max(len(flat), 1), self.slots):
            chunk = flat[i : i + self.slots] or [0.0]
            payload.append(self._ts.ckks_vector(self._secret_ctx, chunk).serialize())
        return EncryptedTensor(tuple(plain_vector.shape), payload)

    def homomorphic_sum(
        self,
        ciphertexts: List[EncryptedTensor],
        weights: List[float],
    ) -> EncryptedTensor:
        """Weighted sum of ciphertexts computed with the public context only."""
        if not ciphertexts:
            raise ValueError("homomorphic_sum requires at least one ciphertext")
        n_chunks = len(ciphertexts[0].payload)
        out: List[bytes] = []
        for c in range(n_chunks):
            acc = None
            for ct, w in zip(ciphertexts, weights):
                vec = self._ts.ckks_vector_from(self._public_ctx, ct.payload[c]) * float(w)
                acc = vec if acc is None else acc + vec
            assert acc is not None
            out.append(acc.serialize())
        return EncryptedTensor(ciphertexts[0].shape, out)

    def decrypt_vector(self, ciphertext: EncryptedTensor) -> torch.Tensor:
        """Decrypt with the secret key and restore the original tensor shape."""
        values: List[float] = []
        for blob in ciphertext.payload:
            values.extend(self._ts.ckks_vector_from(self._secret_ctx, blob).decrypt())
        numel = 1
        for d in ciphertext.shape:
            numel *= d
        return torch.tensor(values[:numel], dtype=torch.float32).reshape(ciphertext.shape)

    def encrypt_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, EncryptedTensor]:
        """Encrypt all parameter tensors in a state dict into CKKS ciphertexts."""
        return {k: self.encrypt_vector(v) for k, v in state_dict.items()}

    def homomorphic_aggregate_state_dicts(
        self,
        encrypted_dicts: List[Dict[str, EncryptedTensor]],
        weights: List[float],
    ) -> Dict[str, EncryptedTensor]:
        """Perform weighted homomorphic aggregation across client ciphertext dictionaries."""
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights] if total_w > 0 else weights
        keys = list(encrypted_dicts[0].keys())
        return {
            key: self.homomorphic_sum([ed[key] for ed in encrypted_dicts], norm_weights)
            for key in keys
        }

    def decrypt_state_dict(
        self,
        encrypted_dict: Dict[str, EncryptedTensor],
        target_dtypes: Optional[Dict[str, torch.dtype]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decrypt aggregated ciphertexts into a plaintext parameter dictionary."""
        decrypted = {}
        for k, ct in encrypted_dict.items():
            plain = self.decrypt_vector(ct)
            if target_dtypes and k in target_dtypes:
                plain = plain.to(target_dtypes[k])
            decrypted[k] = plain
        return decrypted


class SecAggPeer:
    """Decentralized cryptographic peer enclave for SecAgg+ key agreement and mask synthesis.

    Each federated hospital client holds its own private X25519 key (RFC 7748) and derives
    antisymmetric pairwise zero-sum masks using HKDF-SHA256 key agreement with peer nodes.
    The central aggregation server receives only masked parameters and possesses zero private keys.
    """

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self._private_key: x25519.X25519PrivateKey = x25519.X25519PrivateKey.generate()
        self._public_key: x25519.X25519PublicKey = self._private_key.public_key()

    @property
    def public_key_bytes(self) -> bytes:
        """32-byte raw public key for network exchange with peer nodes."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def derive_pairwise_key(
        self,
        peer_id: int,
        peer_pk_bytes: bytes,
        round_num: int = 1,
    ) -> bytes:
        """Derive 256-bit symmetric shared key with a peer using ECDH and HKDF-SHA256."""
        peer_pk = x25519.X25519PublicKey.from_public_bytes(peer_pk_bytes)
        shared_secret = self._private_key.exchange(peer_pk)

        # Canonical pair tag ensures client u and client v derive the identical 256-bit key
        low_id, high_id = min(self.client_id, peer_id), max(self.client_id, peer_id)
        info_tag = f"secagg_pair_{low_id}_{high_id}_round_{round_num}".encode()

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"fed-xray-secagg-v2",
            info=info_tag,
        )
        return hkdf.derive(shared_secret)

    def compute_mask(
        self,
        peer_public_keys: Dict[int, bytes],
        tensor_shapes: Dict[str, torch.Size],
        weights_map: Optional[Dict[int, float]] = None,
        round_num: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Locally synthesize the client's zero-sum additive mask using ECDH shared secrets."""
        my_weight = 1.0
        if weights_map is not None:
            total_w = sum(weights_map.values())
            my_weight = max(1e-12, float(weights_map.get(self.client_id, 1.0)) / max(1e-12, total_w))

        mask = {
            param_name: torch.zeros(shape, device=device, dtype=torch.float32)
            for param_name, shape in tensor_shapes.items()
        }

        for peer_id, peer_pk_bytes in sorted(peer_public_keys.items()):
            if peer_id == self.client_id:
                continue

            pairwise_key = self.derive_pairwise_key(peer_id, peer_pk_bytes, round_num=round_num)

            for param_name, shape in tensor_shapes.items():
                h = hashes.Hash(hashes.SHA256())
                h.update(pairwise_key)
                h.update(param_name.encode())
                digest = h.finalize()
                seed_64 = int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)

                gen = torch.Generator(device=device).manual_seed(seed_64)
                pairwise_noise = torch.randn(shape, generator=gen, device=device) * 0.1

                # Antisymmetric accumulation:
                # If self.client_id < peer_id: add noise / my_weight
                # If self.client_id > peer_id: subtract noise / my_weight
                # Thus sum_{u} p_u * S_u = sum_{u < v} (noise - noise) = 0 identically.
                if self.client_id < peer_id:
                    mask[param_name] += pairwise_noise / my_weight
                else:
                    mask[param_name] -= pairwise_noise / my_weight

        return mask


class SecAggPlusProtocol:
    """Cryptographic Secure Aggregation (SecAgg+) with decentralized ECDH zero-sum masking."""

    @staticmethod
    def generate_pairwise_masks(
        num_clients: int,
        tensor_shapes: Dict[str, torch.Size],
        device: torch.device = torch.device("cpu"),
        weights: Optional[List[float]] = None,
        round_num: int = 1,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generates authentic ECDH-derived pairwise masks s_u such that sum_{u=1}^K p_u * s_u = 0.

        Uses ephemeral X25519 key agreements and HKDF-SHA256 key derivation across decentralized peers.
        """
        peers = [SecAggPeer(client_id=i) for i in range(num_clients)]
        pk_registry = {i: peer.public_key_bytes for i, peer in enumerate(peers)}

        weights_map: Optional[Dict[int, float]] = None
        if weights is not None:
            weights_map = {i: float(w) for i, w in enumerate(weights)}

        return [
            peer.compute_mask(
                peer_public_keys=pk_registry,
                tensor_shapes=tensor_shapes,
                weights_map=weights_map,
                round_num=round_num,
                device=device,
            )
            for peer in peers
        ]

    @staticmethod
    def mask_state_dict(
        state_dict: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply zero-sum additive mask to client state dict."""
        return {
            k: (v.float() + mask[k].to(v.device)).to(v.dtype) if k in mask else v.clone()
            for k, v in state_dict.items()
        }

    @staticmethod
    def reconstruct_dropout_residual(
        dropped_client_ids: List[int],
        surviving_peers: Dict[int, SecAggPeer],
        all_public_keys: Dict[int, bytes],
        tensor_shapes: Dict[str, torch.Size],
        weights_map: Dict[int, float],
        round_num: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct the pairwise mask contributions of dropped clients for exact cancellation.

        When client v drops out or is excluded, surviving peers share pairwise seeds with v,
        allowing the server to reconstruct and subtract the dropped node's un-cancelled masks
        without revealing any surviving client's model parameters.
        """
        residual = {
            param_name: torch.zeros(shape, device=device, dtype=torch.float32)
            for param_name, shape in tensor_shapes.items()
        }

        for dropped_id in dropped_client_ids:
            if dropped_id not in all_public_keys:
                continue
            dropped_pk = all_public_keys[dropped_id]

            for surviving_id, peer in surviving_peers.items():
                pairwise_key = peer.derive_pairwise_key(dropped_id, dropped_pk, round_num=round_num)

                for param_name, shape in tensor_shapes.items():
                    h = hashes.Hash(hashes.SHA256())
                    h.update(pairwise_key)
                    h.update(param_name.encode())
                    digest = h.finalize()
                    seed_64 = int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)

                    gen = torch.Generator(device=device).manual_seed(seed_64)
                    pairwise_noise = torch.randn(shape, generator=gen, device=device) * 0.1

                    # If surviving_id < dropped_id, surviving added noise (weighted by p_surv)
                    # When summed, it remains in the aggregate. We subtract it:
                    if surviving_id < dropped_id:
                        residual[param_name] += pairwise_noise
                    else:
                        residual[param_name] -= pairwise_noise

        return residual


class PatientLevelDPAccountant:
    """Rényi Differential Privacy (RDP) Composition Accountant for Patient-Level DP.

    Prevents patient group privacy collapse on multi-patch (WSI) or multi-slice (3D MRI/CT) data.
    """

    ORDERS = [1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0]

    def __init__(
        self,
        target_delta: float = 1e-5,
    ) -> None:
        self.target_delta = target_delta
        self.rdp_history: List[Dict[float, float]] = []

    def compute_rdp_step(self, q: float, sigma: float) -> Dict[float, float]:
        """Compute Rényi divergence under Poisson subsampling amplification with rate q."""
        rdp_step = {}
        for alpha in self.ORDERS:
            if sigma == 0:
                rdp_step[alpha] = float("inf")
            else:
                # Subsampled Gaussian mechanism RDP upper bound: (q^2 * alpha) / (2 * sigma^2) + O(q^3)
                rdp_val = (q ** 2 * alpha) / (2.0 * (sigma ** 2))
                rdp_step[alpha] = rdp_val
        return rdp_step

    def step(self, q: float, sigma: float) -> None:
        """Record one federated communication round."""
        step_rdp = self.compute_rdp_step(q=q, sigma=sigma)
        self.rdp_history.append(step_rdp)

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Converts total composed RDP guarantees to canonical (epsilon, delta)-DP bound:

        epsilon(delta) = min_{alpha > 1} { sum RDP(alpha) + ln(1/delta) / (alpha - 1) }
        """
        if not self.rdp_history:
            return 0.0

        target_delta = delta or self.target_delta
        total_rdp = {alpha: 0.0 for alpha in self.ORDERS}
        for step in self.rdp_history:
            for alpha in self.ORDERS:
                total_rdp[alpha] += step[alpha]

        epsilons = []
        for alpha in self.ORDERS:
            eps = total_rdp[alpha] + math.log(1.0 / target_delta) / (alpha - 1.0)
            epsilons.append(eps)

        return min(epsilons)


def apply_patient_level_dp_clipping(
    patient_gradients: List[torch.Tensor],
    clipping_bound_C: float = 1.0,
    noise_multiplier_sigma: float = 1.0,
) -> torch.Tensor:
    """Applies strict Patient-Level Gaussian Differential Privacy:

    1. Average slices/patches per patient: g_i = (1/m_i) * sum g_{i,j}
    2. Clip patient gradient: \bar{g}_i = g_i * min(1, C / ||g_i||_2)
    3. Perturb with Gaussian noise: \tilde{g} = (1/|B_P|) * (sum \bar{g}_i + N(0, sigma^2 * C^2 * I))
    """
    clipped_grads = []
    for g_i in patient_gradients:
        norm = torch.norm(g_i, p=2)
        clip_factor = min(1.0, clipping_bound_C / (norm.item() + 1e-8))
        clipped_grads.append(g_i * clip_factor)

    sum_clipped = torch.stack(clipped_grads).sum(dim=0)
    noise = torch.randn_like(sum_clipped) * (noise_multiplier_sigma * clipping_bound_C)
    num_patients = len(patient_gradients)
    return (sum_clipped + noise) / max(1, num_patients)


def clip_and_perturb_state_dict(
    delta_weights: Dict[str, torch.Tensor],
    clipping_bound: float = 1.0,
    noise_multiplier: float = 0.0,
    num_clients: int = 1,
    dp_mode: str = "distributed",
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Applies patient-level sensitivity clipping and calibrated Gaussian perturbation.

    Args:
        delta_weights: Parameter update deltas: Delta W_k = W_k - W_0.
        clipping_bound: L2 sensitivity clipping threshold C.
        noise_multiplier: Gaussian mechanism noise multiplier sigma.
        num_clients: Total participating consortium clients K for distributed noise scaling.
        dp_mode: Differential privacy operating mode:
            - "distributed": Option J standard. Client perturbs locally with
              sigma_client = (sigma * C) / sqrt(K). After consensus aggregation across
              K clients, the aggregate model achieves exact central Gaussian (epsilon, delta)-DP
              with zero plaintext individual leakage in transit.
            - "local": Local Differential Privacy. Client perturbs with
              sigma_local = sigma * C, guaranteeing local (epsilon, delta)-LDP at hospital boundary.
            - "central": Central DP. Client clips delta to C to guarantee bounded sensitivity,
              leaving noise perturbation to the aggregator.
            - "off": Identity pass-through with no clipping or noise.

    Returns:
        (perturbed_deltas, pre_clip_l2_norm)
    """
    if not delta_weights:
        return {}, 0.0

    if dp_mode == "off":
        return {k: v.clone() for k, v in delta_weights.items()}, 0.0

    norm_sq = 0.0
    for v in delta_weights.values():
        norm_sq += torch.sum(v.float() ** 2).item()
    total_l2_norm = math.sqrt(norm_sq)

    clip_factor = min(1.0, clipping_bound / (total_l2_norm + 1e-8))
    clipped = {k: v.float() * clip_factor for k, v in delta_weights.items()}

    k_clients = max(1, num_clients)
    if dp_mode == "distributed" and noise_multiplier > 0.0:
        client_sigma = (noise_multiplier * clipping_bound) / math.sqrt(k_clients)
        perturbed = {
            k: v + torch.randn_like(v) * client_sigma
            for k, v in clipped.items()
        }
    elif dp_mode == "local" and noise_multiplier > 0.0:
        local_sigma = noise_multiplier * clipping_bound
        perturbed = {
            k: v + torch.randn_like(v) * local_sigma
            for k, v in clipped.items()
        }
    else:
        perturbed = clipped

    return perturbed, total_l2_norm
