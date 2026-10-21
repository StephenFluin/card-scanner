import { inject, Service, signal } from '@angular/core';
import { FirebaseService } from './firebase.service';
import { Logger } from './logger.service';

export interface AppUser {
  uid: string;
  displayName: string | null;
  email: string | null;
  photoURL: string | null;
}

/** Firebase Authentication with Google as the only sign-in provider. */
@Service()
export class AuthService {
  private readonly firebase = inject(FirebaseService);
  private readonly logger = inject(Logger);

  readonly enabled = this.firebase.enabled;
  readonly user = signal<AppUser | null>(null);
  /** True once the initial auth state is known. */
  readonly ready = signal(!this.enabled);
  readonly busy = signal(false);
  readonly error = signal<string | null>(null);

  constructor() {
    if (!this.enabled) return;
    void this.init();
  }

  private async init(): Promise<void> {
    const { auth } = await this.firebase.get();
    const { onAuthStateChanged, getRedirectResult } = await import('firebase/auth');
    onAuthStateChanged(auth, (u) => {
      this.user.set(
        u ? { uid: u.uid, displayName: u.displayName, email: u.email, photoURL: u.photoURL } : null,
      );
      this.ready.set(true);
      this.logger.info('auth', u ? `signed in as ${u.email ?? u.uid}` : 'signed out');
    });
    try {
      await getRedirectResult(auth);
    } catch (e) {
      this.fail('redirect sign-in failed', e);
    }
  }

  async signIn(): Promise<void> {
    if (!this.enabled) return;
    this.busy.set(true);
    this.error.set(null);
    const { auth } = await this.firebase.get();
    const { GoogleAuthProvider, signInWithPopup, signInWithRedirect } =
      await import('firebase/auth');
    const provider = new GoogleAuthProvider();
    provider.setCustomParameters({ prompt: 'select_account' });
    try {
      await signInWithPopup(auth, provider);
    } catch (e) {
      const code = (e as { code?: string }).code;
      if (
        code === 'auth/popup-blocked' ||
        code === 'auth/operation-not-supported-in-this-environment'
      ) {
        this.logger.warn('auth', `popup unavailable (${code}), falling back to redirect`);
        await signInWithRedirect(auth, provider);
        return;
      }
      if (code !== 'auth/popup-closed-by-user' && code !== 'auth/cancelled-popup-request') {
        this.fail('sign-in failed', e);
      }
    } finally {
      this.busy.set(false);
    }
  }

  async signOut(): Promise<void> {
    if (!this.enabled) return;
    const { auth } = await this.firebase.get();
    const { signOut } = await import('firebase/auth');
    await signOut(auth);
  }

  private fail(message: string, e: unknown): void {
    const detail = e instanceof Error ? e.message : String(e);
    this.error.set(`${message}: ${detail}`);
    this.logger.error('auth', message, detail);
  }
}
