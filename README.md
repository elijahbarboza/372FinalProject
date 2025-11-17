## System Architecture
```mermaid
flowchart TD
    A[Spotify Web API<br/>(Playlists, Track Metadata,<br/>Audio Features, Preview URLs)]
        --> B[Data Collection & Preprocessing<br/>(Spotipy, Pandas, Feature Engineering)]

    B --> C[Model Training<br/>(Similarity / Ranking Network,<br/>Baseline + Tuned Models)]
    C --> D[Recommendation API<br/>(Flask/FastAPI Endpoint)]

    subgraph WebApp["Front‑End (Streamlit / Gradio)"]
        E[User selects track<br/>or enters vibe query]
        E -->|/recommend?track_id=...| D
        D -->|Top‑N recommendations| F[Display Track Cards<br/>Title, Artist, Score, Play Buttons]
        F --> G[Audio Playback<br/>HTML/Streamlit Audio Player<br/>(30‑s preview_url)]
        F --> H[Optional Mix Endpoint<br/>(/mix?track1&track2)]
    end

    H --> I[Audio Mixing Module<br/>(Beat align + Crossfade with Librosa)]
    I --> G
    F --> J[(Spotify Link)]
    J -->|Open full track in Spotify| X[Spotify App / Web Player]

    style A fill:#1db954,stroke:#0f6b3b,color:#fff
    style B fill:#d1f5d3,stroke:#008040
    style C fill:#cbe5ff,stroke:#0050a0
    style D fill:#b2b2ff,stroke:#4040a0
    style WebApp fill:#ffe9b3,stroke:#a08300
    style I fill:#ffd7b3,stroke:#a04000
    style J fill:#cccccc,stroke:#555