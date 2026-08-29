# PL Nexus — Frontend (React + Vite)

FPL-themed React rebuild of the dashboard, analytics, ML predictions, and PL history pages.
Deep purple (`#37003C`) + signature green (`#00FF87`) on a **white** canvas, with the
recurring "form pip" (W/D/L square) motif tying every page together.

## Local development

```bash
cd frontend-react
npm install
cp .env.example .env      # point VITE_API_BASE_URL at your backend
npm run dev               # http://localhost:5173
```

The backend (FastAPI) must be running and reachable at whatever `VITE_API_BASE_URL` points to.
Locally that's usually `http://localhost:8000` (run `python main.py` from the repo root).

## Deploying to Vercel

1. Push this repo to GitHub.
2. In Vercel: **New Project** → import the repo → set **Root Directory** to `frontend-react`.
3. Vercel auto-detects Vite (`vercel.json` in this folder pins the build command/output anyway).
4. Add an environment variable in the Vercel project settings:
   - `VITE_API_BASE_URL` = your Railway backend URL, e.g. `https://your-app.up.railway.app`
5. Deploy. Every push to your default branch redeploys automatically.

## Deploying the backend to Railway

The repo root already has a `Dockerfile` that Railway can build directly (it reads `$PORT`
automatically, so no changes are needed there). Just:

1. In Railway: **New Project** → **Deploy from GitHub repo** → point it at the repo root (not `frontend-react`).
2. Add your env vars in Railway's dashboard (`DATABASE_URL`, `MLFLOW_TRACKING_URI`,
   `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `FOOTBALL_DATA_KEY`, `NEWS_API_KEY`).
3. Once deployed, copy the Railway-issued URL into the frontend's `VITE_API_BASE_URL`.

## Pages

| Route          | Data source                          | Notes                                    |
|----------------|---------------------------------------|-------------------------------------------|
| `/`            | `GET /api/dashboard`                  | Standings, fixtures, results, news        |
| `/analytics`   | `GET /api/analytics/`                 | 11 leaderboard categories (players/teams) |
| `/predictions` | `GET /api/predictions/`               | ML gameweek predictions + feature importance |
| `/history`     | Static data (`src/data/champions.js`) | PL champions 1992/93–2024/25              |

## Notes

- CORS on the backend is currently wide open (`allow_origins=["*"]`) so it'll work from any
  Vercel preview URL out of the box. Once you're production-stable, consider locking
  `ALLOW_ORIGINS` in `main.py` down to your actual Vercel domain.
- All analytics leaderboard columns are inferred at runtime (name/team/metric), since the
  backend returns raw SQL rows without a fixed schema contract — if a table's shape changes,
  the UI adapts rather than breaking.
