"""RoomEar — ambient audio embedding service.

Captures microphone audio, computes CLAP embeddings, and broadcasts
them via UDP.  No display, no track identification — just:
microphone → CLAP → UDP publish.

Run as:  python3 -m albart.roomear [--device NAME] [--port 57002]
"""
