# CoMLRL Docs (Hugo Book)

This folder contains the Hugo site for CoMLRL’s documentation using the Hugo Book theme.

## Local preview

1. Install Hugo Extended: https://gohugo.io/installation/
2. From the repo root, run:

```bash
hugo server -s docs -D
```

Hugo will fetch the Book theme via Hugo Modules on first run (requires internet access). Open the local URL printed by Hugo (usually http://localhost:1313).

## Deploy (GitHub Pages – native actions)

Pushing to `main` runs `.github/workflows/docs.yml`, which:

- Configures Pages and computes the correct base URL for a project site.
- Builds the Hugo site from `docs/`.
- Uploads the static site as a Pages artifact and deploys it.

One-time repo setting: in GitHub → Settings → Pages → Build and deployment, set Source to “GitHub Actions”.

> Note: GitHub Pages sites are publicly accessible even if the repository is private. For private/internal docs, preview locally or host behind authentication (e.g., GitHub Enterprise Pages with SSO or another internal host).
