import asyncio
import os
import tempfile
import edge_tts


async def _make_tts(text: str, output_path: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice="ko-KR-SunHiNeural",
    )
    await communicate.save(output_path)


def play_tts(text: str):
    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name

    try:
        asyncio.run(_make_tts(text, output_path))

        os.system(f'ffplay -nodisp -autoexit -loglevel quiet "{output_path}"')

    finally:
        if os.path.exists(output_path):
            os.remove(output_path)