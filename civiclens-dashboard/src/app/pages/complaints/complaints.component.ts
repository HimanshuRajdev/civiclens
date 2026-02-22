import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Complaint } from '../../models/complaint.model';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-complaints',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">Complaint Management</h1>
          <p class="page-subtitle">View, filter, and manage all reported civic infrastructure issues</p>
        </div>
        <div class="header-stats">
          <div class="header-stat">
            <span class="hs-value">{{ complaints.length }}</span>
            <span class="hs-label">Total</span>
          </div>
          <div class="header-stat open">
            <span class="hs-value">{{ getCount('Open') }}</span>
            <span class="hs-label">Open</span>
          </div>
          <div class="header-stat progress">
            <span class="hs-value">{{ getCount('In Progress') }}</span>
            <span class="hs-label">Active</span>
          </div>
          <div class="header-stat resolved">
            <span class="hs-value">{{ getCount('Resolved') }}</span>
            <span class="hs-label">Resolved</span>
          </div>
        </div>
      </div>

      <!-- Filters -->
      <div class="filters-bar">
        <div class="search-box">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" [(ngModel)]="searchQuery" (ngModelChange)="filterComplaints()" placeholder="Search by ID, type, or description..." class="filter-input" />
        </div>
        <select [(ngModel)]="filterStatus" (ngModelChange)="filterComplaints()" class="input filter-select">
          <option value="">All Status</option>
          <option value="Open">Open</option>
          <option value="In Progress">In Progress</option>
          <option value="Resolved">Resolved</option>
        </select>
        <select [(ngModel)]="filterType" (ngModelChange)="filterComplaints()" class="input filter-select">
          <option value="">All Types</option>
          <option *ngFor="let t of issueTypes" [value]="t">{{ t }}</option>
        </select>
        <select [(ngModel)]="filterSeverity" (ngModelChange)="filterComplaints()" class="input filter-select">
          <option value="">All Severity</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <!-- Table -->
      <div class="card table-card">
        <table class="data-table" *ngIf="filteredComplaints.length > 0">
          <thead>
            <tr>
              <th>ID</th>
              <th>Issue Type</th>
              <th>Description</th>
              <th>Severity</th>
              <th>Department</th>
              <th>Status</th>
              <th>Reported</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let c of filteredComplaints; let i = index"
                class="table-row"
                [class.expanded]="expandedId === c.id"
                [style.animation-delay]="i * 0.03 + 's'">
              <td>
                <span class="complaint-id">{{ c.id }}</span>
              </td>
              <td>
                <div class="type-cell">
                  <span class="type-dot" [ngClass]="'dot-' + getTypeColor(c.issue_type)"></span>
                  {{ c.issue_type }}
                </div>
              </td>
              <td>
                <span class="desc-text">{{ c.description | slice:0:50 }}{{ c.description.length > 50 ? '...' : '' }}</span>
              </td>
              <td>
                <span class="badge" [ngClass]="'badge-' + c.severity.toLowerCase()">{{ c.severity }}</span>
              </td>
              <td><span class="dept-text">{{ c.department }}</span></td>
              <td>
                <span class="badge" [ngClass]="{
                  'badge-open': c.status === 'Open',
                  'badge-in-progress': c.status === 'In Progress',
                  'badge-resolved': c.status === 'Resolved'
                }">{{ c.status }}</span>
              </td>
              <td><span class="time-text">{{ formatDate(c.created_at) }}</span></td>
              <td>
                <div class="action-btns">
                  <button class="btn btn-sm btn-warning" *ngIf="c.status === 'Open'" (click)="updateStatus(c, 'In Progress')" title="Start Processing">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  </button>
                  <button class="btn btn-sm btn-success" *ngIf="c.status === 'In Progress'" (click)="updateStatus(c, 'Resolved')" title="Mark Resolved">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
                  </button>
                  <button class="btn btn-sm btn-secondary" (click)="toggleExpand(c.id)" title="Details">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/></svg>
                  </button>
                </div>
              </td>
            </tr>
            <!-- Expanded Detail Row -->
            <tr *ngIf="expandedId && expandedComplaint" class="detail-row">
              <td colspan="8">
                <div class="detail-content">
                  <div class="detail-image" *ngIf="expandedComplaint.image_url">
                    <img [src]="getImageUrl(expandedComplaint.image_url)" alt="Complaint Image" />
                  </div>
                  <div class="detail-info">
                    <div class="detail-field">
                      <label>Full Description</label>
                      <p>{{ expandedComplaint.description }}</p>
                    </div>
                    <div class="detail-grid">
                      <div class="detail-field">
                        <label>Location</label>
                        <p>{{ expandedComplaint.lat.toFixed(4) }}, {{ expandedComplaint.lng.toFixed(4) }}</p>
                      </div>
                      <div class="detail-field">
                        <label>Department</label>
                        <p>{{ expandedComplaint.department }}</p>
                      </div>
                      <div class="detail-field">
                        <label>Filed At</label>
                        <p>{{ expandedComplaint.created_at }}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>

        <div class="empty-state" *ngIf="filteredComplaints.length === 0 && !loading">
          <div class="empty-icon">🔍</div>
          <h4>No complaints found</h4>
          <p>Try adjusting your filters or search query</p>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
    }

    .header-stats {
      display: flex;
      gap: 16px;
    }

    .header-stat {
      text-align: center;
      padding: 8px 16px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);

      .hs-value { display: block; font-size: 20px; font-weight: 800; color: var(--text-primary); }
      .hs-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }

      &.open { border-color: rgba(239, 68, 68, 0.3); .hs-value { color: var(--danger); } }
      &.progress { border-color: rgba(245, 158, 11, 0.3); .hs-value { color: var(--warning); } }
      &.resolved { border-color: rgba(16, 185, 129, 0.3); .hs-value { color: var(--success); } }
    }

    .filters-bar {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
    }

    .search-box {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0 14px;
      flex: 1;
      color: var(--text-muted);

      &:focus-within {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px var(--accent-blue-dim);
      }
    }

    .filter-input {
      background: none;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-size: 13px;
      font-family: var(--font-family);
      padding: 10px 0;
      width: 100%;
      &::placeholder { color: var(--text-muted); }
    }

    .filter-select {
      width: 160px;
      flex-shrink: 0;
    }

    .table-card {
      padding: 0;
      overflow: hidden;
    }

    .table-row {
      animation: fadeIn 0.3s ease forwards;
      opacity: 0;
    }

    .complaint-id {
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 12px;
      color: var(--accent-blue);
      font-weight: 600;
    }

    .type-cell {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 500;
      color: var(--text-primary);
    }

    .type-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      &.dot-blue { background: var(--accent-blue); }
      &.dot-purple { background: var(--accent-purple); }
      &.dot-cyan { background: var(--accent-cyan); }
      &.dot-warning { background: var(--warning); }
      &.dot-danger { background: var(--danger); }
      &.dot-success { background: var(--success); }
    }

    .desc-text {
      font-size: 12px;
      color: var(--text-secondary);
    }

    .dept-text {
      font-size: 12px;
      color: var(--text-secondary);
    }

    .time-text {
      font-size: 12px;
      color: var(--text-muted);
      white-space: nowrap;
    }

    .action-btns {
      display: flex;
      gap: 4px;
    }

    /* Detail row */
    .detail-row td {
      padding: 0 !important;
      border-bottom: 2px solid var(--accent-blue) !important;
    }

    .detail-content {
      display: flex;
      gap: 20px;
      padding: 20px;
      background: var(--bg-surface-2);
      animation: slideUp 0.2s ease;
    }

    .detail-image {
      width: 200px;
      height: 150px;
      border-radius: var(--radius-md);
      overflow: hidden;
      flex-shrink: 0;

      img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
    }

    .detail-info {
      flex: 1;
    }

    .detail-field {
      margin-bottom: 12px;

      label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        font-weight: 600;
        display: block;
        margin-bottom: 4px;
      }

      p {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.5;
      }
    }

    .detail-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
  `]
})
export class ComplaintsComponent implements OnInit {
  complaints: Complaint[] = [];
  filteredComplaints: Complaint[] = [];
  loading = true;
  searchQuery = '';
  filterStatus = '';
  filterType = '';
  filterSeverity = '';
  expandedId: string | null = null;
  expandedComplaint: Complaint | null = null;

  issueTypes = ['Pothole', 'Sinkhole', 'Water Leakage', 'Garbage Overflow', 'Broken Streetlight', 'Broken Sidewalk'];

  private typeColors: Record<string, string> = {
    'Pothole': 'blue',
    'Sinkhole': 'danger',
    'Water Leakage': 'cyan',
    'Garbage Overflow': 'warning',
    'Broken Streetlight': 'purple',
    'Broken Sidewalk': 'success',
  };

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadComplaints();
  }

  loadComplaints() {
    this.api.getComplaints().subscribe({
      next: (c) => {
        this.complaints = c;
        this.filterComplaints();
        this.loading = false;
      },
      error: () => {
        this.complaints = [];
        this.filteredComplaints = [];
        this.loading = false;
      },
    });
  }

  filterComplaints() {
    let result = [...this.complaints];
    if (this.filterStatus) result = result.filter(c => c.status === this.filterStatus);
    if (this.filterType) result = result.filter(c => c.issue_type === this.filterType);
    if (this.filterSeverity) result = result.filter(c => c.severity === this.filterSeverity);
    if (this.searchQuery) {
      const q = this.searchQuery.toLowerCase();
      result = result.filter(c =>
        c.id.toLowerCase().includes(q) ||
        c.issue_type.toLowerCase().includes(q) ||
        c.description.toLowerCase().includes(q) ||
        c.department.toLowerCase().includes(q)
      );
    }
    this.filteredComplaints = result;
  }

  updateStatus(c: Complaint, newStatus: string) {
    this.api.updateStatus(c.id, newStatus).subscribe({
      next: () => {
        c.status = newStatus as any;
        this.filterComplaints();
      },
    });
  }

  toggleExpand(id: string) {
    if (this.expandedId === id) {
      this.expandedId = null;
      this.expandedComplaint = null;
    } else {
      this.expandedId = id;
      this.expandedComplaint = this.complaints.find(c => c.id === id) || null;
    }
  }

  getImageUrl(url: string): string {
    if (url.startsWith('http')) return url;
    return `${environment.apiUrl}${url}`;
  }

  getTypeColor(type: string): string {
    return this.typeColors[type] || 'blue';
  }

  getCount(status: string): number {
    return this.complaints.filter(c => c.status === status).length;
  }

  formatDate(dateStr: string): string {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
}
