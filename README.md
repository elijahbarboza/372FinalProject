## System Architecture
```mermaid
flowchart TD
    A[Raw Training Data\n(GPS splits, cadence, HR, temp, demographics)] 
        --> B[Data Preprocessing\n(cleaning, normalization, feature engineering)]
    B --> C1[Baseline ML Models\n(Linear Reg., XGBoost, Random Forest)]
    B --> C2[Deep Time‑Series Models\n(LSTM, Transformer Encoder)]
    C1 --> D[Performance Prediction – race time, fatigue index]
    C2 --> D
    D --> E[Evaluation & Visualization\n(R^2, RMSE, training curves, error plots)]
    D --> F[Workout Recommendation Layer\n(Reinforcement learning / heuristic policy)]
    F --> G[Next‑Day Workout Suggestion\n(easy run, tempo, rest)]
    style A fill:#d9f5ff,stroke:#0077b6
    style B fill:#e6ffe6,stroke:#228B22
    style C1 fill:#f9f9b5,stroke:#b59d00
    style C2 fill:#f9f9b5,stroke:#b59d00
    style D fill:#ffcccc,stroke:#b30000
    style E fill:#d9c2ff,stroke:#6a0dad
    style F fill:#ffe6cc,stroke:#cc6600
    style G fill:#fff2cc,stroke:#d4a017