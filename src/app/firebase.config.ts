import type { FirebaseOptions } from 'firebase/app';

/**
 * Firebase web app config. Get it from the Firebase console:
 * Project settings -> General -> Your apps -> Web app -> SDK setup and configuration.
 *
 * Also enable:
 *  - Authentication -> Sign-in method -> Google
 *  - Realtime Database (then deploy database.rules.json: `firebase deploy --only database`)
 *  - Authentication -> Settings -> Authorized domains: add any LAN host you test from
 *
 * While `apiKey` is empty the app runs in local-only mode (collection kept in
 * this browser's localStorage, sign-in hidden).
 */
export const firebaseConfig: FirebaseOptions = {
  apiKey: '',
  authDomain: '',
  databaseURL: '',
  projectId: '',
  storageBucket: '',
  messagingSenderId: '',
  appId: '',
};

export const firebaseConfigured = !!firebaseConfig.apiKey && !!firebaseConfig.databaseURL;
