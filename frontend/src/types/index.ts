export interface HospitalCohort {
  hospital_id: number;
  name: string;
  num_samples: number;
  distribution: number[];
  counts: {
    normal: number;
    pneumonia: number;
    covid: number;
  };
  sample_images: number[][][];
  sample_labels: number[];
}

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
}

export interface DiagnosisResult {
  predicted_class: number;
  predicted_name: string;
  true_class: number;
  true_name: string;
  confidence: number;
  probabilities: number[];
  findings: string;
  raw_image: number[][];
  heatmap: number[][];
}

export interface RagCase {
  case_id: string;
  label_id: number;
  label_name: string;
  similarity: number;
  history: string;
}
