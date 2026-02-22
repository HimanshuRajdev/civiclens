import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';
import { Complaint, Stats, DepartmentStats, TimelinePoint } from '../models/complaint.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getStats(): Observable<Stats> {
    return this.http.get<Stats>(`${this.baseUrl}/stats`);
  }

  getComplaints(status?: string, issueType?: string): Observable<Complaint[]> {
    let params = new HttpParams();
    if (status) params = params.set('status', status);
    if (issueType) params = params.set('issue_type', issueType);
    return this.http.get<Complaint[]>(`${this.baseUrl}/complaints`, { params });
  }

  getComplaintById(id: string): Observable<Complaint> {
    return this.http.get<Complaint>(`${this.baseUrl}/complaints/${id}`);
  }

  updateStatus(id: string, status: string): Observable<any> {
    return this.http.patch(`${this.baseUrl}/complaints/${id}/status`, { status });
  }

  getTimeline(): Observable<TimelinePoint[]> {
    return this.http.get<TimelinePoint[]>(`${this.baseUrl}/stats/timeline`);
  }

  getDepartmentStats(): Observable<DepartmentStats[]> {
    return this.http.get<DepartmentStats[]>(`${this.baseUrl}/stats/departments`);
  }

  getResolutionStats(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/stats/resolution`);
  }

  deleteComplaint(id: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/complaints/${id}`);
  }
}
