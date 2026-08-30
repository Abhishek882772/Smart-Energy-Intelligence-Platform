# SmartEnergy

## Local development on Windows

Use Node.js 20 or newer. From the project directory, install dependencies with:

```powershell
npm install
```

Start the full-stack development server with:

```powershell
npm run dev
```

The development scripts use `cross-env`, so the `NODE_ENV` assignment works in PowerShell, Command Prompt, macOS, and Linux. If an older local checkout still contains `@builder.io/vite-plugin-jsx-loc`, remove that package or pull the latest project files; it is not required for the app and is incompatible with Vite 7.

The project also supports the package-manager-native command:

```powershell
pnpm install
pnpm dev
```
