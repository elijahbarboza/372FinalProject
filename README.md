## System Architecture
```mermaid
flowchart TD
    A[Spotify Web API – playlists, track metadata, audio features, preview URLs]
        --> B[Data collection and preprocessing – Spotipy, Pandas, feature engineering]
    B --> C[Model training – similarity or ranking network, baseline and tuned models]
    C --> D[Recommendation API – Flask or FastAPI endpoint]
    subgraph WebApp["Front‑End – Streamlit or Gradio"]
        E[User selects track or enters vibe query]
        E -->|/recommend?track_id=...| D
        D -->|Top‑N recommendations| F[Display track cards – title, artist, score, play buttons]
        F --> G[Audio playback – Streamlit audio player for 30‑second preview_url]
        F --> H[Optional mix endpoint – /mix?track1&track2]
    end
    H --> I[Audio mixing module – beat alignment and cross‑fade with Librosa]
    I --> G
    F --> J[Spotify link]
    J -->|Open full track in Spotify| X[Spotify app or web player]

    style A fill:#1db954,stroke:#0f6b3b,color:#fff
    style B fill:#d1f5d3,stroke:#008040
    style C fill:#cbe5ff,stroke:#0050a0
    style D fill:#b2b2ff,stroke:#4040a0
    style WebApp fill:#ffe9b3,stroke:#a08300
    style I fill:#ffd7b3,stroke:#a04000
    style J fill:#cccccc,stroke:#555