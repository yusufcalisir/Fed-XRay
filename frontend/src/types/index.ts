export interface HospitalCohort {
  hospital_id: number;
  name: string;
  num_samples: number;
  distribution: number[];
  counts: {
    normal?: number;
    pneumonia?: number;
    covid?: number;
    nevus?: number;
    melanoma?: number;
    carcinoma?: number;
    benign_dysplastic_nevus?: number;
    malignant_melanoma?: number;
    basal_cell_carcinoma?: number;
    actinic_keratosis?: number;
    benign_keratosis?: number;
    dermatofibroma?: number;
    vascular_lesion?: number;
    squamous_cell_carcinoma?: number;
    [key: string]: number | undefined;
  };
  sample_images: number[][][];
  sample_labels: number[];
  provenance?: string;
  is_authentic_clinical?: boolean;
}

export interface FederatedTrainingOptions {
  numRounds?: number;
  localEpochs?: number;
  learningRate?: number;
  privacyNoise?: number;
  simulateAttack?: boolean;
  activateDefense?: boolean;
  algorithm?: string;
  lossFn?: string;
  modelType?: string;
  peftMode?: string;
  secureAggregation?: boolean;
  useCkks?: boolean;
  pretrained?: boolean;
  dpMode?: string;
  dpClippingBound?: number;
  strictCheckpoint?: boolean;
}

export type TrainingConfig = FederatedTrainingOptions;

export interface TelemetryRound {
  round_num: number;
  total_rounds: number;
  train_loss: number;
  train_accuracy: number;
  test_loss: number;
  test_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  threat_detected: boolean;
  blocked_nodes: number[];
  status: string;
  model_type?: string;
  peft_mode?: string;
  privacy_epsilon?: number;
  uplink_bytes_per_client?: number;
  full_model_bytes?: number;
}

export interface DiagnosisResult {
  patient_id?: string;
  predicted_class: number;
  predicted_name: string;
  true_class: number;
  true_name: string;
  confidence: number;
  probabilities: number[];
  findings: string;
  raw_image: number[][];
  heatmap: number[][] | null;
  heatmap_available?: boolean;
  heatmap_error?: string | null;
  image_source?: string;
  provenance?: string;
  anatomical_site?: string;
  biopsy_confirmed?: boolean;
  modality?: string;
}

export interface RagCase {
  case_id: string;
  label_id: number;
  label_name: string;
  similarity: number;
  history: string;
  biopsy_confirmed?: boolean;
  therapeutic_outcome?: string;
  anatomical_site?: string;
  site_id?: string;
  provenance?: string;
}

export interface HospitalNodeTelemetry {
  id: string;
  name: string;
  location: string;
  scanner: string;
  scans: number;
  modalityKey: string;
  skewKey: string;
  adapter: string;
  dpEpsilon: number;
  status: "online" | "syncing";
}

export interface BackendSystemState {
  model_trained: boolean;
  hospitals_loaded: number;
  has_test_set: boolean;
  history_rounds: number;
  model_type: string;
  peft_mode: string;
  persisted_on_disk: boolean;
  data_source: string;
  data_provenance?: string;
  is_authentic_clinical?: boolean;
  strict_real_data?: boolean;
  cohorts?: HospitalCohort[];
  persistence?: {
    backend: string;
    db_path: string;
    stats: {
      db_path: string;
      wal_mode_enabled: boolean;
      size_bytes: number;
      rounds_count: number;
      cohorts_count: number;
      audits_count: number;
      checkpoints_count: number;
    };
  };
  privacy: {
    dp_active: boolean;
    dp_mode: string;
    epsilon: number;
    target_delta: number;
    target_bound: string;
  };
  hospitals: HospitalNodeTelemetry[];
  rounds: {
    round_num: number;
    train_loss: number;
    train_accuracy: number;
    test_loss?: number;
    test_accuracy?: number;
    f1_score: number;
    bandwidth: string;
    threat_detected: boolean;
    status: string;
  }[];
  reference_cases: {
    case_id: string;
    diagnosis: string;
    similarity: number;
    clinical_notes: string;
    site_id?: string;
    biopsy_confirmed?: boolean;
    anatomical_site?: string;
    provenance?: string;
  }[];
  is_live: boolean;
  telemetry_source: "live_runtime" | "reference_benchmark";
}

export interface EnclaveNodeInfo {
  node_id: string;
  name: string;
  location: string;
  enclave_status: string;
  peer_key_fingerprint: string;
  scans_verified: number;
  peft_subspace: string;
  leakage_bytes: number;
  enclave_type?: string;
}

export interface CryptographySuiteInfo {
  homomorphic_encryption: string;
  secagg_protocol: string;
  key_exchange: string;
  cipher_ring: string;
}

export interface EnclaveHandshakeResponse {
  attestation_status: string;
  attestation_digest: string;
  timestamp: string;
  enclave_architecture?: string;
  is_hardware_tee?: boolean;
  server_enclave: {
    status: string;
    device: string;
    cuda_available: boolean;
    defense_mode: string;
    foundation_model: string;
    model_trained?: boolean;
    enclave_type?: string;
    threshold_mpc_status?: string;
  };
  nodes: EnclaveNodeInfo[];
  cryptography: CryptographySuiteInfo;
  privacy: {
    dp_active?: boolean;
    patient_level_dp?: boolean;
    epsilon: number;
    target_delta: number;
    clipping_bound_C?: number;
    mechanism?: string;
  };
}
