## System Architecture
```mermaid
flowchart TD
    A[Raw training data – GPS splits, cadence, heart rate, temperature, demographics] 
        --> B[Data preprocessing and feature engineering – cleaning, normalization, rolling averages]
    B --> C1[Baseline models – linear regression, XGBoost, random forest]
    B --> C2[Deep time-series models – LSTM, Transformer encoder]
    C1 --> D[Performance prediction – race time or fatigue index]
    C2 --> D
    D --> E[Evaluation and visualization – R2, RMSE, error plots, training curves]
    D --> F[Workout recommendation layer – reinforcement or rule-based policy]
    F --> G[Next-day workout suggestion – easy run, tempo, rest]
    style A fill:#d9f5ff,stroke:#0077b6,color:#000
    style B fill:#e6ffe6,stroke:#228B22,color:#000
    style C1 fill:#f9f9b5,stroke:#b59d00,color:#000
    style C2 fill:#f9f9b5,stroke:#b59d00,color:#000
    style D fill:#ffcccc,stroke:#b30000,color:#000
    style E fill:#d9c2ff,stroke:#6a0dad,color:#000
    style F fill:#ffe6cc,stroke:#cc6600,color:#000
    style G fill:#fff2cc,stroke:#d4a017,color:#000