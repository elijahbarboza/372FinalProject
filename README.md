## System Architecture
```mermaid
flowchart TD
    A[Spotify Web API\n(Playlists, Track Metadata,\nAudio Features, Preview URLs)]
        --> B[Data Collection & Preprocessing\n(Spotipy, Pandas, Feature Engineering)]
    B --> C[Model Training\n(Similarity / Ranking Network,\nBaseline + Tuned Models)]
    C --> D[Recommendation API\n(Flask/FastAPI Endpoint)]
    subgraph WebApp["Front‑End (Streamlit / Gradio)"]
        E[User selects track\nor enters vibe query]
        E -->|/recommend?track_id=...| D
        D -->|Top‑N recommendations| F[Display Track Cards\nTitle, Artist, Score, Play Buttons]
        F --> G[Audio Playback\nHTML/Streamlit Audio Player\n(30‑s preview_url)]
        F --> H[Optional Mix Endpoint\n(/mix?track1&track2)]
    end
    H --> I[Audio Mixing Module\n(Beat align + Crossfade with Librosa)]
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