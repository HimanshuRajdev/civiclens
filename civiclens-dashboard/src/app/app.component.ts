import { Component } from '@angular/core';
import { RouterOutlet, Router, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { HeaderComponent } from './components/header/header.component';
import { filter, map } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, SidebarComponent, HeaderComponent],
  template: `
    <div class="app-shell">
      <app-sidebar></app-sidebar>
      <div class="main-area">
        <app-header [pageTitle]="pageTitle"></app-header>
        <main class="main-content">
          <router-outlet></router-outlet>
        </main>
      </div>
    </div>
  `,
  styles: [`
    .app-shell {
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    .main-area {
      flex: 1;
      margin-left: var(--sidebar-width);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .main-content {
      flex: 1;
      overflow: hidden;
    }
  `]
})
export class AppComponent {
  pageTitle = 'Overview';

  private titleMap: Record<string, string> = {
    '/dashboard': 'Overview',
    '/complaints': 'Complaints',
    '/analytics': 'Analytics',
    '/map': 'Live Map',
    '/ai-insights': 'AI Insights',
  };

  constructor(private router: Router) {
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        map(e => e.urlAfterRedirects)
      )
      .subscribe(url => {
        this.pageTitle = this.titleMap[url] || 'Overview';
      });
  }
}
