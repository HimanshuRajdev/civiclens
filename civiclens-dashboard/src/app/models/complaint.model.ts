export interface Complaint {
  id: string;
  filename: string;
  image_url: string;
  issue_type: string;
  severity: string;
  description: string;
  department: string;
  lat: number;
  lng: number;
  status: 'Open' | 'In Progress' | 'Resolved';
  created_at: string;
}

export interface Stats {
  total: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
  by_severity: Record<string, number>;
}

export interface TimelinePoint {
  date: string;
  count: number;
}

export interface DepartmentStats {
  department: string;
  total: number;
  open: number;
  in_progress: number;
  resolved: number;
}

export interface DetectionResult {
  class: string;
  confidence: number;
  severity: string;
  department: string;
}

export interface ReportResponse {
  image_url: string;
  filename: string;
  detection: DetectionResult;
  complaint: {
    title: string;
    description: string;
    action_required: string;
    priority: string;
    department: string;
  };
  location: { lat: number; lng: number };
  duplicate: {
    is_duplicate: boolean;
    existing_id?: string;
    distance_meters?: number;
  };
}
