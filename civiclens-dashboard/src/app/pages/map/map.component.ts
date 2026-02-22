import { Component, OnInit, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Complaint } from '../../models/complaint.model';
import * as L from 'leaflet';

@Component({
  selector: 'app-map-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="map-wrapper">
      <!-- Map Controls Overlay -->
      <div class="map-controls">
        <div class="control-panel">
          <h3 class="panel-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/></svg>
            Incident Map
          </h3>
          <div class="filter-group">
            <label>Filter by Status</label>
            <select [(ngModel)]="filterStatus" (ngModelChange)="applyFilters()" class="input">
              <option value="">All</option>
              <option value="Open">Open</option>
              <option value="In Progress">In Progress</option>
              <option value="Resolved">Resolved</option>
            </select>
          </div>
          <div class="filter-group">
            <label>Filter by Severity</label>
            <select [(ngModel)]="filterSeverity" (ngModelChange)="applyFilters()" class="input">
              <option value="">All</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
          </div>
          <div class="filter-group">
            <label>Filter by Type</label>
            <select [(ngModel)]="filterType" (ngModelChange)="applyFilters()" class="input">
              <option value="">All Types</option>
              <option *ngFor="let t of issueTypes" [value]="t">{{ t }}</option>
            </select>
          </div>

          <div class="legend">
            <span class="legend-title">Severity Legend</span>
            <div class="legend-item"><span class="legend-dot high"></span> High</div>
            <div class="legend-item"><span class="legend-dot medium"></span> Medium</div>
            <div class="legend-item"><span class="legend-dot low"></span> Low</div>
          </div>

          <div class="map-stats">
            <div class="map-stat">
              <span class="ms-value">{{ displayedCount }}</span>
              <span class="ms-label">Visible</span>
            </div>
            <div class="map-stat">
              <span class="ms-value">{{ complaints.length }}</span>
              <span class="ms-label">Total</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Map Container -->
      <div id="mapContainer" class="map-container"></div>
    </div>
  `,
  styles: [`
    .map-wrapper {
      position: relative;
      height: calc(100vh - var(--header-height));
      overflow: hidden;
    }

    .map-container {
      width: 100%;
      height: 100%;
      background: var(--bg-primary);
    }

    .map-controls {
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 1000;
    }

    .control-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 20px;
      width: 240px;
      backdrop-filter: blur(10px);
    }

    .panel-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 15px;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 16px;
    }

    .filter-group {
      margin-bottom: 12px;

      label {
        display: block;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 4px;
      }

      .input {
        padding: 7px 10px;
        font-size: 12px;
      }
    }

    .legend {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--border-light);
    }

    .legend-title {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.3px;
      display: block;
      margin-bottom: 8px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      &.high { background: #ef4444; box-shadow: 0 0 6px rgba(239, 68, 68, 0.5); }
      &.medium { background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.5); }
      &.low { background: #3b82f6; box-shadow: 0 0 6px rgba(59, 130, 246, 0.5); }
    }

    .map-stats {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--border-light);
    }

    .map-stat {
      flex: 1;
      text-align: center;
      .ms-value { display: block; font-size: 18px; font-weight: 800; color: var(--text-primary); }
      .ms-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    }

    :host ::ng-deep {
      .leaflet-tile-pane { filter: brightness(0.7) contrast(1.1) saturate(0.3) hue-rotate(180deg); }
      .leaflet-control-zoom a {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
      }
      .custom-popup .leaflet-popup-content-wrapper {
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        box-shadow: var(--shadow-lg);
      }
      .custom-popup .leaflet-popup-tip { background: var(--bg-surface); }
    }
  `]
})
export class MapPageComponent implements OnInit, AfterViewInit, OnDestroy {
  complaints: Complaint[] = [];
  filteredComplaints: Complaint[] = [];
  filterStatus = '';
  filterSeverity = '';
  filterType = '';
  displayedCount = 0;
  issueTypes = ['Pothole', 'Sinkhole', 'Water Leakage', 'Garbage Overflow', 'Broken Streetlight', 'Broken Sidewalk'];

  private map!: L.Map;
  private markersLayer!: L.LayerGroup;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getComplaints().subscribe({
      next: (c) => {
        this.complaints = c;
        this.filteredComplaints = c;
        this.displayedCount = c.length;
        this.plotMarkers();
      },
      error: () => {
        this.complaints = [];
      },
    });
  }

  ngAfterViewInit() {
    this.initMap();
  }

  ngOnDestroy() {
    if (this.map) this.map.remove();
  }

  private initMap() {
    this.map = L.map('mapContainer', {
      center: [43.0731, -89.4012],
      zoom: 13,
      zoomControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap',
      maxZoom: 19,
    }).addTo(this.map);

    this.markersLayer = L.layerGroup().addTo(this.map);
  }

  applyFilters() {
    let result = [...this.complaints];
    if (this.filterStatus) result = result.filter(c => c.status === this.filterStatus);
    if (this.filterSeverity) result = result.filter(c => c.severity === this.filterSeverity);
    if (this.filterType) result = result.filter(c => c.issue_type === this.filterType);
    this.filteredComplaints = result;
    this.displayedCount = result.length;
    this.plotMarkers();
  }

  private plotMarkers() {
    if (!this.markersLayer) return;
    this.markersLayer.clearLayers();

    const severityColors: Record<string, string> = {
      'High': '#ef4444',
      'Medium': '#f59e0b',
      'Low': '#3b82f6',
    };

    for (const c of this.filteredComplaints) {
      if (!c.lat || !c.lng || (c.lat === 0 && c.lng === 0)) continue;

      const color = severityColors[c.severity] || '#3b82f6';
      const icon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="
          width: 14px; height: 14px;
          background: ${color};
          border-radius: 50%;
          border: 2px solid rgba(255,255,255,0.8);
          box-shadow: 0 0 10px ${color}80;
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      const marker = L.marker([c.lat, c.lng], { icon });

      const statusBadge = c.status === 'Open'
        ? '<span style="background:rgba(239,68,68,0.2);color:#ef4444;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600">Open</span>'
        : c.status === 'In Progress'
        ? '<span style="background:rgba(245,158,11,0.2);color:#f59e0b;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600">In Progress</span>'
        : '<span style="background:rgba(16,185,129,0.2);color:#10b981;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600">Resolved</span>';

      marker.bindPopup(`
        <div style="min-width:200px;font-family:Inter,sans-serif;">
          <div style="font-weight:700;font-size:13px;margin-bottom:4px">${c.issue_type}</div>
          <div style="font-size:11px;color:#8585a0;margin-bottom:8px">${c.id}</div>
          <div style="font-size:12px;color:#b0b0c0;margin-bottom:8px;line-height:1.4">${c.description.slice(0, 100)}${c.description.length > 100 ? '...' : ''}</div>
          <div style="display:flex;justify-content:space-between;align-items:center">
            ${statusBadge}
            <span style="font-size:10px;color:#55556a">${c.department}</span>
          </div>
        </div>
      `, { className: 'custom-popup' });

      marker.addTo(this.markersLayer);
    }

    // Fit bounds if we have markers
    const validComplaints = this.filteredComplaints.filter(c => c.lat && c.lng && !(c.lat === 0 && c.lng === 0));
    if (validComplaints.length > 0) {
      const bounds = L.latLngBounds(validComplaints.map(c => [c.lat, c.lng] as [number, number]));
      this.map.fitBounds(bounds, { padding: [50, 50] });
    }
  }
}
