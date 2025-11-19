## System Architecture
```mermaid
flowchart TD
    A[Open MIDI datasets – Lakh, MAESTRO, BitMidi]
        --> B[Data preprocessing – parse notes, durations, velocities, quantize timing, create token sequences]
    B --> C[Automatic Sonic Pi target generation – convert MIDI events to Sonic Pi token templates]
    C --> D[Dataset pairing – aligned input and target token sequences stored as JSON or CSV]
    D --> E[Model training – encoder‑decoder Transformer with cross‑entropy loss and validation monitoring]
    E --> F[Evaluation and analysis – BLEU accuracy, loss curves, error inspection, audio playback tests]
    F --> G[Code reconstruction – detokenize model output into executable Sonic Pi Ruby code]
    G --> H[Sonic Pi engine – play generated music via OSC or save .rb script]
    F --> I[Streamlit or Gradio app – upload MIDI, generate Sonic Pi code, preview audio]

    style A fill:#d9f5ff,stroke:#0077b6,color:#000
    style B fill:#e6ffe6,stroke:#228B22,color:#000
    style C fill:#fff6cc,stroke:#b59d00,color:#000
    style D fill:#f0e6ff,stroke:#6a0dad,color:#000
    style E fill:#cce0ff,stroke:#0050a0,color:#000
    style F fill:#ffd9d9,stroke:#b30000,color:#000
    style G fill:#ffe6cc,stroke:#cc6600,color:#000
    style H fill:#ffc299,stroke:#a04000,color:#000
    style I fill:#f9f9f9,stroke:#555,color:#000