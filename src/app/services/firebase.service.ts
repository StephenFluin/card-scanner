import { isPlatformBrowser } from '@angular/common';
import { inject, PLATFORM_ID, Service } from '@angular/core';
import type { Auth } from 'firebase/auth';
import type { Database } from 'firebase/database';
import { firebaseConfig, firebaseConfigured } from '../firebase.config';
import { Logger } from './logger.service';

export interface FirebaseHandles {
  auth: Auth;
  db: Database;
}

/** Lazily initializes Firebase (browser only) so it stays out of the initial bundle. */
@Service()
export class FirebaseService {
  private readonly logger = inject(Logger);
  private handles: Promise<FirebaseHandles> | null = null;

  /** False when no config is filled in or when running on the server. */
  readonly enabled = firebaseConfigured && isPlatformBrowser(inject(PLATFORM_ID));

  get(): Promise<FirebaseHandles> {
    if (!this.enabled) return Promise.reject(new Error('Firebase is not configured'));
    this.handles ??= (async () => {
      const [{ initializeApp }, { getAuth }, { getDatabase }] = await Promise.all([
        import('firebase/app'),
        import('firebase/auth'),
        import('firebase/database'),
      ]);
      const app = initializeApp(firebaseConfig);
      this.logger.info('app', `Firebase initialized (project ${firebaseConfig.projectId})`);
      return { auth: getAuth(app), db: getDatabase(app) };
    })();
    return this.handles;
  }
}
