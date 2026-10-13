import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    title: 'Scan · Card Scanner',
    loadComponent: () => import('./scanner/scanner-page').then((m) => m.ScannerPage),
  },
  {
    path: 'collection',
    title: 'Collection · Card Scanner',
    loadComponent: () => import('./collection/collection-page').then((m) => m.CollectionPage),
  },
  { path: '**', redirectTo: '' },
];
