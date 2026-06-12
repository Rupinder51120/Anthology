%% Anthology v2 RAG System Architecture
%% High-level data flow and component relationships

graph TB
    subgraph "Data Sources"
        PDFs["📄 PDF Files<br/>data/papers/"]
        ArXiv["🔗 ArXiv API<br/>Paper Discovery"]
    end

    subgraph "Ingestion Pipeline"
        Parse["PyMuPDF Parsing<br/>load_paper"]
        Chunk["Intelligent Chunking<br/>chunk_paper<br/>1400-1800 chars/chunk"]
        Filter["Quality Filtering<br/>score_chunk<br/>min_score=0.3"]
        Embed["BGE Embedding<br/>BAAI/bge-large-en-v1.5<br/>384-dim vectors"]
    end

    subgraph "Index Layer"
        NumpyIdx["NumpyIndex<br/>faiss_index.bin"]
        BM25Idx["BM25 Index<br/>bm25_index.pkl"]
        EncIdx["Embeddings<br/>chunk_embeddings.npy"]
        DB["PostgreSQL + pgvector<br/>chunks table"]
    end

    subgraph "Retrieval Pipeline"
        Query["User Query"]
        QEmbed["Embed Query<br/>embed_texts"]
        Dense["pgvector Search<br/>top-15 dense"]
        Lexical["PostgreSQL FTS<br/>top-15 lexical"]
        RRF["RRF Fusion<br/>K=60<br/>top-10 merged"]
        Rerank["Cross-Encoder<br/>ms-marco-MiniLM<br/>top-5 final"]
    end

    subgraph "Generation Pipeline"
        CtxFmt["Format Context<br/>format_context"]
        LLM["Ollama LLM<br/>qwen2.5:7b<br/>OR Groq (cloud)"]
        CitFmt["Extract Citations<br/>format_citations"]
        Memory["Conversation Memory<br/>ConversationMemory"]
    end

    subgraph "Evaluation Framework"
        Retrieval["Retrieval Metrics<br/>recall@k, MRR, NDCG"]
        Generation["Generation Metrics<br/>faithfulness, relevancy<br/>context_precision"]
        Pipeline["Pipeline Runner<br/>run_pipeline_on_dataset"]
    end

    subgraph "API Layer"
        API["FastAPI Server<br/>api/main.py"]
        QRouter["Query Router<br>/api/v1/query"]
        SRouter["Search Router<br/>/api/v1/search"]
        RRouter["Recommend Router<br/>/api/v1/recommend"]
        PRouter["Papers Router<br/>/api/v1/papers"]
    end

    subgraph "UI Layer"
        Streamlit["Streamlit App<br/>app.py"]
    end

    subgraph "Scripts"
        BuildIdx["Build Index<br/>scripts/build_index.py"]
        Benchmark["Benchmark<br/>scripts/run_benchmark.py"]
    end

    %% Data flow
    PDFs --> Parse
    ArXiv --> Download["arxiv_downloader"]
    Download --> PDFs

    Parse --> Chunk
    Chunk --> Filter
    Filter --> Embed

    Embed --> NumpyIdx
    Embed --> EncIdx
    Filter --> BM25Idx
    Parse --> DB

    %% Retrieval flow
    Query --> QEmbed
    QEmbed --> Dense
    QEmbed --> Lexical
    Dense --> RRF
    Lexical --> RRF
    RRF --> Rerank

    %% Generation flow
    Rerank --> CtxFmt
    CtxFmt --> LLM
    LLM --> CitFmt
    CitFmt --> Memory

    %% API flow
    QRouter --> Rerank
    Rerank --> LLM
    SRouter --> Rerank
    RRouter --> Memory
    PRouter --> DB

    API --> QRouter
    API --> SRouter
    API --> RRouter
    API --> PRouter

    %% UI flow
    Streamlit --> API
    Streamlit --> Memory

    %% Evaluation flow
    Pipeline --> Retrieval
    Pipeline --> Generation
    Benchmark --> Pipeline

    %% Index building
    BuildIdx --> Filter
    BuildIdx --> Embed
    BuildIdx --> NumpyIdx
    BuildIdx --> BM25Idx

    %% Styling
    classDef dataSource fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef ingestion fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef index fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef retrieval fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef generation fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef evaluation fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef api fill:#ede7f6,stroke:#311b92,stroke-width:2px
    classDef ui fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef scripts fill:#e0f2f1,stroke:#00695c,stroke-width:2px

    class PDFs,ArXiv,Download dataSource
    class Parse,Chunk,Filter,Embed ingestion
    class NumpyIdx,BM25Idx,EncIdx,DB index
    class Query,QEmbed,Dense,Lexical,RRF,Rerank retrieval
    class CtxFmt,LLM,CitFmt,Memory generation
    class Retrieval,Generation,Pipeline evaluation
    class API,QRouter,SRouter,RRouter,PRouter api
    class Streamlit ui
    class BuildIdx,Benchmark scripts
