"""C1 Explanation (XAI) agent: template text from the engine's steps, an optional LLM
plain-language rewrite, and a deterministic post-check that rejects any changed number or
citation. Never imports an LLM SDK; the LLM client is passed in.
"""
