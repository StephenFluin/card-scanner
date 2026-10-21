# CardScanner

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 22.2.2.

## Development server

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Vitest](https://vitest.dev/) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Working on this project

These conventions are for anyone changing the code, whether that's a person or an AI coding
assistant working in the repo.

### Before you commit

```bash
npm run lint           # ESLint, including Angular template and accessibility rules
npm run format         # Prettier (CI runs npm run format:check)
npm run build
```

### Angular and TypeScript style

The app follows current Angular (v22) conventions:

- **Components** are standalone (the default, so don't add `standalone: true`), small, and do one
  job. OnPush change detection is the default, so don't set `changeDetection`. Small components use
  inline templates; external template and style paths are relative to the component file.
- **State** lives in signals: `signal()` for local state, `computed()` for derived state, and
  `linkedSignal()` for state that must follow several sources. Change signals with `set()` or
  `update()` and keep transformations pure.
- **Inputs and outputs** use `input()`, `output()` and, for two-way bindings, `model()` rather
  than decorators. Inject dependencies with `inject()`, and make singletons
  `providedIn: 'root'` (or `@Service` for new ones).
- **Templates** use built-in control flow (`@if`, `@for`, `@switch`), `class` and `style` bindings
  instead of `ngClass`/`ngStyle`, the async pipe for observables, and import only the directives
  and pipes they use (no `CommonModule`). Host bindings go in the decorator's `host` object, not
  `@HostBinding`/`@HostListener`. Keep logic out of templates, and don't reach for globals such as
  `new Date()` there.
- **Forms**: Signal Forms (`@angular/forms/signals`) for new forms, otherwise reactive forms.
- **Images**: `NgOptimizedImage` for static images (it can't handle inline base64 images).
- **Routing**: lazy-load feature routes.
- **TypeScript** is strict. Let inference work when the type is obvious, and use `unknown` rather
  than `any`.

### Accessibility

Every page must pass AXE checks and meet WCAG AA, including focus management, color contrast and
ARIA attributes.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.

## License

Copyright Stephen Fluin

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>. The full text is
in [LICENSE](LICENSE).
