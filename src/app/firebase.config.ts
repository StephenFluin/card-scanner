import type { FirebaseOptions } from 'firebase/app';

/**
 * Firebase web app config (project card-scanner-234235). These values are
 * public identifiers, not secrets: access is enforced by database.rules.json
 * and Authentication's authorized domains.
 *
 * Refresh with: firebase apps:sdkconfig WEB <appId>
 * Database rules: firebase deploy --only database
 * Testing from another host (e.g. a phone on the LAN)? Add it under
 * Authentication -> Settings -> Authorized domains.
 *
 * With an empty `apiKey` the app runs in local-only mode (collection kept in
 * this browser's localStorage, sign-in hidden).
 */
export const firebaseConfig: FirebaseOptions = {
  apiKey: 'AIzaSyB9WFg_YltlCMBSHIdzxfbviT2--J-RhsU',
  authDomain: 'card-scanner-234235.firebaseapp.com',
  databaseURL: 'https://card-scanner-234235-default-rtdb.firebaseio.com',
  projectId: 'card-scanner-234235',
  storageBucket: 'card-scanner-234235.firebasestorage.app',
  messagingSenderId: '1031793483667',
  appId: '1:1031793483667:web:303cc713300a21de089266',
};

export const firebaseConfigured = !!firebaseConfig.apiKey && !!firebaseConfig.databaseURL;
