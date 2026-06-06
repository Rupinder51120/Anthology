import re
import subprocess
import tempfile
import shutil
from pathlib import Path


# Voice options on macOS: Samantha (default), Alex, Victoria, Karen, Daniel, Moira
TTS_VOICE  = "Samantha"
TTS_RATE   = 175  # words per minute — natural pace


def clean_text_for_speech(text: str) -> str:
    if not text:
        return ""

    # Remove code blocks entirely
    text = re.sub(r'```[\s\S]*?```', ' ', text)

    # Remove math blocks
    text = re.sub(r'\[MATH_BLOCK\][\s\S]*?\[/MATH_BLOCK\]', ' ', text)
    text = re.sub(r'\[MATH\][\s\S]*?\[/MATH\]', ' ', text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', ' ', text)
    text = re.sub(r'\$[^\$]+?\$', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', ' ', text)

    # Remove citation markers
    text = re.sub(r'\[Source\s+\d+\]', '', text)
    text = re.sub(r'\(\s*Source\s+\d+\s*\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    # Section headers — speak them naturally
    text = re.sub(r'\*\*EXPLANATION\*\*', 'Explanation.', text)
    text = re.sub(r'\*\*KEY INSIGHT\*\*', 'Key insight.', text)
    text = re.sub(r'\*\*FROM THE PAPERS\*\*', 'From the papers.', text)

    # Remove remaining markdown
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'^[ \t]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold → plain
    text = re.sub(r'\*(.+?)\*',     r'\1', text)  # italic → plain
    text = re.sub(r'__(.+?)__',     r'\1', text)
    text = re.sub(r'_(.+?)_',       r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)  # inline code → plain

    # Expand abbreviations for natural speech
    replacements = {
        'e.g.':  'for example',
        'i.e.':  'that is',
        'et al.': 'and others',
        'Fig.':  'Figure',
        'Eq.':   'Equation',
        ' vs ':  ' versus ',
        ' vs. ': ' versus ',
        'ReLU':  'Relu activation',
        'GANs':  'Generative Adversarial Networks',
        'LLM':   'large language model',
        'LLMs':  'large language models',
        'RAG':   'retrieval augmented generation',
        'NLP':   'natural language processing',
        'CNN':   'convolutional neural network',
        'RNN':   'recurrent neural network',
        'LSTM':  'long short-term memory network',
    }
    for abbr, expansion in replacements.items():
        text = text.replace(abbr, expansion)

    # Math symbol substitutions
    text = text.replace('=',  ' equals ')
    text = text.replace('≈',  ' approximately ')
    text = text.replace('≠',  ' not equal to ')
    text = text.replace('≤',  ' less than or equal to ')
    text = text.replace('≥',  ' greater than or equal to ')
    text = text.replace('→',  ' leads to ')
    text = text.replace('←',  ' comes from ')
    text = text.replace('∑',  ' sum of ')
    text = text.replace('∏',  ' product of ')
    text = text.replace('∈',  ' in ')
    text = text.replace('∉',  ' not in ')
    text = text.replace('⊂',  ' subset of ')
    text = text.replace('%',  ' percent')

    # Clean up whitespace
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)

    return text.strip()


def text_to_speech(text: str, voice: str = TTS_VOICE, rate: int = TTS_RATE) -> bytes | None:
    if not text:
        return None

    say_path = shutil.which("say")
    if not say_path:
        print("Warning: macOS 'say' command not found.")
        return None

    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return None

    # Limit to ~4000 chars for reasonable audio length
    cleaned = cleaned[:4000]

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_wav = Path(tmpdir) / "speech.wav"
        cmd = [
            "say",
            "-v", voice,
            "-r", str(rate),
            "-o", str(temp_wav),
            "--data-format=LEI16@22050",
            cleaned
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if temp_wav.exists():
                return temp_wav.read_bytes()
            print(f"TTS Error: WAV not created. Stderr: {result.stderr}")
            return None
        except subprocess.CalledProcessError as e:
            # Fallback: try default voice if specified voice fails
            if voice != "Samantha":
                return text_to_speech(text, voice="Samantha", rate=rate)
            print(f"TTS Exception: {e}")
            return None
        except Exception as e:
            print(f"TTS Exception: {e}")
            return None