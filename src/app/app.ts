import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './services/auth.service';
import { CollectionService } from './services/collection.service';

@Component({
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  selector: 'app-root',
  template: `
    <a class="skip-link" href="#main">Skip to content</a>
    <header class="app-header">
      <a class="brand" routerLink="/">Card Scanner</a>
      <nav aria-label="Main">
        <a
          routerLink="/"
          routerLinkActive="active"
          [routerLinkActiveOptions]="{ exact: true }"
          ariaCurrentWhenActive="page"
          >Scan</a
        >
        <a routerLink="/collection" routerLinkActive="active" ariaCurrentWhenActive="page">
          Collection <span class="badge">{{ collection.totalCards() }}</span>
        </a>
      </nav>
      <div class="account">
        @if (auth.enabled) {
          @if (auth.user(); as user) {
            <span class="user" [title]="user.email ?? ''">{{
              user.displayName ?? user.email
            }}</span>
            <button type="button" (click)="auth.signOut()">Sign out</button>
          } @else if (auth.ready()) {
            <button type="button" class="primary" (click)="auth.signIn()" [disabled]="auth.busy()">
              Sign in with Google
            </button>
          }
        } @else {
          <span
            class="badge warn"
            title="Fill in src/app/firebase.config.ts to enable sign-in and cloud sync"
            >Local only</span
          >
        }
      </div>
    </header>
    @if (auth.error(); as err) {
      <p class="banner" role="alert">{{ err }}</p>
    }
    @if (auth.enabled && auth.ready() && !auth.user() && collection.totalCards() > 0) {
      <p class="banner">
        Cards you scan are saved on this device. Sign in to keep them in your account.
      </p>
    }
    <main id="main">
      <router-outlet />
    </main>
  `,
  styles: `
    .skip-link {
      position: absolute;
      left: -999px;
    }
    .skip-link:focus {
      left: 0.5rem;
      top: 0.5rem;
      z-index: 10;
      background: var(--surface);
      padding: 0.5rem;
    }
    .app-header {
      display: flex;
      align-items: center;
      gap: 1rem;
      min-height: var(--header-h);
      padding: 0.25rem 1rem;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-wrap: wrap;
    }
    .brand {
      font-weight: 700;
      color: var(--accent);
      text-decoration: none;
    }
    nav {
      display: flex;
      gap: 0.25rem;
    }
    nav a {
      color: var(--text);
      text-decoration: none;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
    }
    nav a.active {
      background: var(--surface-2);
      box-shadow: inset 0 -2px 0 var(--accent);
    }
    .account {
      margin-left: auto;
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .user {
      max-width: 12rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text-muted);
    }
    .banner {
      margin: 0;
      padding: 0.5rem 1rem;
      background: #2a2410;
      border-bottom: 1px solid #4a3d10;
    }
    main {
      padding: 0.75rem;
      max-width: 1200px;
      margin: 0 auto;
    }
  `,
})
export class App {
  protected readonly auth = inject(AuthService);
  protected readonly collection = inject(CollectionService);
}
