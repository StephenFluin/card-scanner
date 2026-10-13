import { RenderMode, ServerRoute } from '@angular/ssr';

export const serverRoutes: ServerRoute[] = [
  {
    // Camera, auth and storage are browser-only; render everything on the client.
    path: '**',
    renderMode: RenderMode.Client,
  },
];
