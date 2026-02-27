#!/usr/bin/env python3
"""
BSL to Speech Pipeline (Direction 1)

Converts recognized BSL glosses to spoken English.

Pipeline:
    Video Features → BSL Recognizer → Glosses → Text Processor → TTS → Audio
"""

import re
from typing import List, Optional
from pathlib import Path


class GlossToText:
    """
    Convert BSL glosses to English text.

    Modes:
    - 'simple': Basic concatenation with minor fixes
    - 'llm': Local FLAN-T5 gloss→English (requires transformers)
    """

    def __init__(self, mode: str = "simple", api_key: str = None):
        self.mode = mode
        self.api_key = api_key

        self.replacements = {
            "i": "I",
            "im": "I'm",
            "dont": "don't",
            "cant": "can't",
            "wont": "won't",
            "didnt": "didn't",
            "isnt": "isn't",
            "arent": "aren't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "youre": "you're",
            "theyre": "they're",
            "hes": "he's",
            "shes": "she's",
            "its": "it's",
            "whats": "what's",
            "thats": "that's",
            "whos": "who's",
            "lets": "let's",
        }

    def process(self, glosses: List[str], max_glosses: int = 50) -> str:
        if not glosses:
            return ""

        glosses = glosses[:max_glosses]

        if self.mode == "llm":
            return self._process_llm(glosses)
        return self._process_simple(glosses)

    def _process_simple(self, glosses: List[str]) -> str:
        words = []
        for gloss in glosses:
            g = (gloss or "").strip()
            if not g:
                continue
            gloss_lower = g.lower()
            words.append(self.replacements.get(gloss_lower, gloss_lower))

        text = " ".join(words).strip()
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _process_llm(self, glosses: List[str]) -> str:
        try:
            return self._call_local_llm(glosses)
        except Exception as e:
            print(f"LLM processing failed: {e}")
            print("Falling back to simple mode")
            return self._process_simple(glosses)

    def _call_local_llm(self, glosses: List[str]) -> str:
        try:
            from transformers import pipeline

            if not hasattr(self, "_llm_pipeline"):
                print("Loading LLM for gloss-to-text (first time only)...")
                self._llm_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=0 if self._has_cuda() else -1,
                )

            gloss_str = " ".join(glosses)
            prompt = (
                "Translate these British Sign Language gloss tokens into natural English. "
                "Do not add information not present in the glosses. "
                f"Glosses: {gloss_str}"
            )

            result = self._llm_pipeline(prompt, max_length=120, do_sample=False)
            text = (result[0].get("generated_text") or "").strip()

            if text and text[-1] not in ".!?":
                text += "."
            return text

        except ImportError:
            raise Exception("transformers library not installed. Run: pip install transformers")

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False


class GlossToTextWithGroq:
    """
    Convert BSL glosses to English using Groq API.
    Key upgrades:
    - Cleans noisy gloss streams (dedupe, cap length, optional stopwords)
    - Strong anti-hallucination prompt
    - temperature=0 for stability
    - Dynamic output length (short glosses => 1 sentence; longer => up to 3 sentences)
    """

    def __init__(
        self,
        api_key: str = None,
        *,
        max_len_short: int = 12,
        max_len_long: int = 30,
        short_sentence_cutoff: int = 12,
        use_stop_glosses: bool = True,
        enable_normalize_map: bool = False,
    ):
        import os

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: No Groq API key. Set GROQ_API_KEY or pass api_key parameter.")

        # Tuning knobs
        self.max_len_short = max_len_short
        self.max_len_long = max_len_long
        self.short_sentence_cutoff = short_sentence_cutoff
        self.use_stop_glosses = use_stop_glosses
        self.enable_normalize_map = enable_normalize_map

        # Optional: remove frequent “fluff” tokens that often create rambling summaries
        # (You can tune this list based on your outputs.)
        self.stop_glosses = {
            "best",
            "great",
            "good",
            "right",
            "everything",
            "thing",
            "world",
            "beautiful",
            "love",
            "happy",
            "dream",
            "change",
            "global",
            "international",
        }

        # Optional: normalization for common recognizer confusions (OFF by default)
        self.normalize_map = {
            # "century": "centre",  # enable only if you’re confident this is a common error in your data
        }

    def process(self, glosses: List[str], max_glosses: int = 50) -> str:
        if not glosses:
            return ""

        if not self.api_key:
            return self._simple_fallback(glosses[: min(max_glosses, 50)])

        # Decide whether this is “short” or “long/noisy” input
        raw = glosses[: min(max_glosses, 50)]
        is_short = len(raw) <= self.short_sentence_cutoff

        # Clean aggressively but safely
        cleaned = self._clean_glosses(
            raw,
            max_len=self.max_len_short if is_short else self.max_len_long,
        )
        gloss_str = " ".join(cleaned).strip()
        if not gloss_str:
            return self._simple_fallback(raw[:10])

        # Dynamic output constraints
        if is_short:
            output_rule = "Output exactly 1 short sentence."
            max_tokens = 60
        else:
            output_rule = "Output up to 3 short sentences (max ~60 words total)."
            max_tokens = 140

        system_prompt = (
            "You translate British Sign Language (BSL) gloss tokens into natural English.\n"
            "Rules:\n"
            "- Do NOT add details, facts, names, locations, motives, or events that are not explicitly in the glosses.\n"
            "- Do NOT 'expand' into a story.\n"
            "- If the glosses are ambiguous, translate literally and minimally.\n"
            f"- {output_rule}\n"
            "- Use simple everyday English.\n"
            "Return ONLY the English output (no explanations)."
        )

        user_prompt = f"Glosses: {gloss_str}"

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )

            text = (response.choices[0].message.content or "").strip()
            text = self._post_clean_text(text, is_short=is_short)

            # If the model returned nothing useful, fall back
            return text if text else self._simple_fallback(cleaned)

        except ImportError:
            print("groq library not installed. Run: pip install groq")
            return self._simple_fallback(cleaned)
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._simple_fallback(cleaned)

    def _clean_glosses(self, glosses: List[str], max_len: int) -> List[str]:
        out = []
        prev = None

        for g in glosses:
            g = (g or "").strip().lower()
            if not g:
                continue

            if self.enable_normalize_map:
                g = self.normalize_map.get(g, g)

            # Drop consecutive duplicates
            if g == prev:
                continue
            prev = g

            # Optional stopword removal
            if self.use_stop_glosses and g in self.stop_glosses:
                continue

            # Keep only reasonable tokens (avoid weird artifacts)
            if not re.fullmatch(r"[a-z']+", g):
                continue

            out.append(g)
            if len(out) >= max_len:
                break

        # If we removed too much, fall back to just dedupe + cap (no stopwords)
        if len(out) < 3:
            out = []
            prev = None
            for g in glosses:
                g = (g or "").strip().lower()
                if not g:
                    continue
                if self.enable_normalize_map:
                    g = self.normalize_map.get(g, g)
                if g == prev:
                    continue
                prev = g
                if re.fullmatch(r"[a-z']+", g):
                    out.append(g)
                if len(out) >= max_len:
                    break

        return out

    def _post_clean_text(self, text: str, *, is_short: bool) -> str:
        # Collapse whitespace, strip quotes
        text = text.replace("\n", " ").strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return ""

        if is_short:
            # Keep only the first sentence to enforce “1 sentence”
            parts = re.split(r"(?<=[.!?])\s+", text)
            text = (parts[0] if parts else text).strip()

        # Ensure ending punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _simple_fallback(self, glosses: List[str]) -> str:
        text = " ".join((g or "").strip().lower() for g in glosses if (g or "").strip())
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text += "."
        return text


class TextToSpeech:
    """
    Convert text to speech audio.

    Supports multiple TTS backends:
    - coqui (XTTS v2 - high quality, custom voice cloning)
    - pyttsx3 (offline, cross-platform, lower quality)
    - gtts (Google, requires internet)
    """

    def __init__(self, engine: str = "coqui", speaker_wav: str = None, language: str = "en"):
        self.engine_name = engine
        self.language = language
        self.speaker_wav = speaker_wav
        self.engine = None
        self.tts_model = None

        if engine == "coqui":
            self._init_coqui()
        elif engine == "pyttsx3":
            self._init_pyttsx3()
        elif engine == "gtts":
            self._init_gtts()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")

    def _init_coqui(self):
        try:
            from TTS.api import TTS
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self.device = device
            print(f"Coqui TTS initialized on {device}")
        except ImportError:
            print("Coqui TTS not installed. Run: pip install TTS")
            raise

    def _init_pyttsx3(self):
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.9)
        except ImportError:
            print("pyttsx3 not installed. Run: pip install pyttsx3")
            raise

    def _init_gtts(self):
        try:
            from gtts import gTTS

            self.gtts = gTTS
        except ImportError:
            print("gTTS not installed. Run: pip install gtts")
            raise

    def speak(self, text: str) -> None:
        if not text:
            return

        if self.engine_name == "coqui":
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                self.save(text, f.name)
                if os.name == "nt":
                    import winsound

                    winsound.PlaySound(f.name, winsound.SND_FILENAME)
                else:
                    os.system(f"aplay {f.name} 2>/dev/null || afplay {f.name}")
                os.unlink(f.name)

        elif self.engine_name == "pyttsx3":
            self.engine.say(text)
            self.engine.runAndWait()

        elif self.engine_name == "gtts":
            import tempfile
            import os

            tts = self.gtts(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                if os.name == "nt":
                    os.system(f"start {f.name}")
                else:
                    os.system(f"mpg123 {f.name} 2>/dev/null || afplay {f.name}")

    def save(self, text: str, output_path: str) -> str:
        if not text:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine_name == "coqui":
            if self.speaker_wav is None:
                raise ValueError("speaker_wav required for Coqui TTS. Provide reference voice file.")

            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=self.speaker_wav,
                language=self.language,
                file_path=str(output_path),
            )

        elif self.engine_name == "pyttsx3":
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()

        elif self.engine_name == "gtts":
            tts = self.gtts(text=text, lang="en")
            tts.save(str(output_path))

        return str(output_path)


class BSLToSpeechPipeline:
    """
    Complete BSL to Speech pipeline:
    - BSL Recognition (glosses from video)
    - Gloss to Text (glosses to English)
    - Text to Speech (English to audio)
    """

    def __init__(
        self,
        recognizer=None,
        tts_engine: str = "coqui",
        speaker_wav: str = None,
        gloss_mode: str = "simple",
        groq_api_key: str = None,
    ):
        self.recognizer = recognizer

        if gloss_mode == "groq":
            self.gloss_to_text = GlossToTextWithGroq(api_key=groq_api_key)
        else:
            self.gloss_to_text = GlossToText(mode=gloss_mode)

        try:
            self.tts = TextToSpeech(engine=tts_engine, speaker_wav=speaker_wav)
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")
            self.tts = None

    def process_features(
        self,
        feature_path: str,
        window_stride: float = 1.0,
        confidence_threshold: float = 0.5,
        speak: bool = True,
        save_audio: str = None,
    ) -> dict:
        if self.recognizer is None:
            raise ValueError("Recognizer not initialized")

        glosses = self.recognizer.recognize_sequence(
            feature_path=feature_path,
            window_stride=window_stride,
            confidence_threshold=confidence_threshold,
        )

        text = self.gloss_to_text.process(glosses)

        audio_path = None
        if self.tts:
            if save_audio:
                audio_path = self.tts.save(text, save_audio)
            if speak:
                self.tts.speak(text)

        return {"glosses": glosses, "text": text, "audio_path": audio_path}

    def process_glosses(self, glosses: List[str], speak: bool = True, save_audio: str = None) -> dict:
        text = self.gloss_to_text.process(glosses)

        audio_path = None
        if self.tts:
            if save_audio:
                audio_path = self.tts.save(text, save_audio)
            if speak:
                self.tts.speak(text)

        return {"glosses": glosses, "text": text, "audio_path": audio_path}


def create_pipeline(
    model_path: str = "outputs/best_model.pt",
    vocab_path: str = "data/processed/vocabulary.json",
    model_type: str = "transformer",
    tts_engine: str = "coqui",
    speaker_wav: str = "data/processed/voice_training.wav",
) -> BSLToSpeechPipeline:
    from inference.recognizer import BSLRecognizer

    recognizer = BSLRecognizer(model_path=model_path, vocab_path=vocab_path, model_type=model_type)

    return BSLToSpeechPipeline(
        recognizer=recognizer,
        tts_engine=tts_engine,
        speaker_wav=speaker_wav,
    )
