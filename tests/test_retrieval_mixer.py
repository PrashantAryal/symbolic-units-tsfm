import numpy as np

from symtsfm.symbolic.retrieval_mixer import AdaptiveRetrievalMixer, SymbolicEmbeddingRetriever


def test_symbolic_reranking_changes_a_tied_embedding_retrieval():
    # Both historical contexts are equally close in embedding space. The
    # symbolic event selects the future trajectory belonging to the query.
    emb = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
    sym = np.array([[[0.]], [[1.]]], dtype=np.float32)
    future = np.array([[[1.]], [[-1.]]], dtype=np.float32)
    r = SymbolicEmbeddingRetriever(n_neighbors=1, symbolic_weight=1.0).fit(emb, sym, future)
    q = r.retrieve(np.array([[1., 0.]], dtype=np.float32), np.array([[[1.]]], dtype=np.float32))
    assert q.indices[0, 0] == 1
    assert q.future[0, 0, 0, 0] == -1.0


def test_adaptive_mixer_can_learn_to_use_a_helpful_retrieved_future():
    n = 24
    emb = np.stack([np.linspace(0, 1, n), np.ones(n)], axis=1).astype(np.float32)
    sym = emb[:, None, :]
    target = emb[:, :1, None]
    base = np.zeros_like(target)
    retriever = SymbolicEmbeddingRetriever(n_neighbors=1, symbolic_weight=0.).fit(emb, sym, target)
    got = retriever.retrieve(emb, sym)
    mixer = AdaptiveRetrievalMixer(2, 1, 1, hidden_dim=8, epochs=80, patience=12, lr=1e-2, device="cpu", seed=4)
    mixer.fit(base[:16], emb[:16], RetrievalSubset(got, slice(0, 16)), target[:16],
              base[16:], emb[16:], RetrievalSubset(got, slice(16, None)), target[16:])
    out, gate = mixer.predict(base[16:], emb[16:], RetrievalSubset(got, slice(16, None)))
    assert np.mean((out - target[16:]) ** 2) < np.mean((base[16:] - target[16:]) ** 2)
    assert gate.mean() > 0


def RetrievalSubset(batch, idx):
    return type(batch)(*(getattr(batch, k)[idx] for k in ("future", "embedding_score", "symbolic_score", "weights", "indices")))
