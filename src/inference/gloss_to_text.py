"""
Gloss to Text Converter for Direction 1 (BSL -> Speech)

Converts BSL glosses to natural English sentences.
BSL has different grammar from English:
- Topic-comment structure
- Time markers at beginning
- No articles (a, an, the)
- Different word order

Modes:
- simple: Basic concatenation (fast, lower quality)
- llm: Local FLAN-T5 model
- groq: Hosted LLM API (best quality). Tries Cerebras gpt-oss-120b first,
  falls back to Groq llama-3.3-70b-versatile if Cerebras fails or times out.
"""

import os
import re
from typing import List, Optional, Dict

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GlossToText:
    """
    Convert BSL glosses to natural English text.
    """
    
    # Common BSL grammar patterns for rule-based conversion
    QUESTION_WORDS = {'what', 'where', 'when', 'who', 'why', 'how', 'which'}
    TIME_MARKERS = {'yesterday', 'today', 'tomorrow', 'now', 'later', 'before', 
                    'after', 'morning', 'afternoon', 'evening', 'night',
                    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                    'saturday', 'sunday', 'week', 'month', 'year'}
    
    def __init__(
        self,
        mode: str = "simple",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize converter.
        
        Args:
            mode: Conversion mode ('simple', 'llm', 'groq')
            groq_api_key: API key for Groq (required if mode='groq')
            groq_model: Groq model to use (fallback provider)
        """
        self.mode = mode
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.groq_model = groq_model
        # Cerebras is the primary provider in 'groq' mode; Groq is the fallback.
        self.cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
        self.cerebras_model = "gpt-oss-120b"
        self.cerebras_timeout = 10  # seconds; keep short so a hang falls back fast
        
        # Built on first use rather than here. Cerebras is the primary provider
        # and speaks plain HTTP, so it must not be gated behind the fallback's
        # SDK being installed.
        self.groq_client = None

        if mode == 'llm':
            self._init_local_llm()
        elif mode == 'groq':
            self._init_hosted()
    
    def _init_local_llm(self):
        """Initialize local FLAN-T5 model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("Transformers required. Install with: pip install transformers")
        
        print("Loading FLAN-T5 model...")
        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_model.to(self.device)
        print(f"FLAN-T5 loaded on {self.device}")
    
    def _init_hosted(self):
        """
        Check that at least one hosted provider is usable, without demanding
        both.

        Cerebras is primary and needs only an API key and requests. Groq is the
        fallback and needs its SDK installed as well. Requiring the Groq SDK up
        front meant a machine without it could not use Cerebras at all, even
        with a valid Cerebras key, because construction failed before the
        primary was ever tried.

        Only a setup with no usable provider is an error, and it says which
        piece is missing rather than naming one provider.
        """
        cerebras_ok = bool(self.cerebras_api_key)
        groq_ok = self._init_groq_client()

        if cerebras_ok:
            print(f"Cerebras ready (model: {self.cerebras_model}, primary)")
        if groq_ok:
            print(f"Groq ready (model: {self.groq_model}, fallback)")

        if not cerebras_ok and not groq_ok:
            raise ValueError(
                "No hosted provider is usable. Set CEREBRAS_API_KEY for the "
                "primary provider, or set GROQ_API_KEY and install the Groq "
                "SDK (pip install groq) for the fallback."
            )

        if not cerebras_ok:
            print("[LLM] No Cerebras key; the Groq fallback will serve every request")
        if not groq_ok:
            print("[LLM] Groq fallback unavailable; a Cerebras failure will fall "
                  "through to rule-based output")

    def _init_groq_client(self) -> bool:
        """
        Build the Groq client if it can be built. Returns whether it is usable.

        Never raises: a missing key or a missing SDK simply means no fallback,
        which is reported by the caller rather than being fatal.
        """
        if self.groq_client is not None:
            return True
        if not self.groq_api_key:
            return False
        try:
            from groq import Groq
        except ImportError:
            return False
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
        except Exception:
            self.groq_client = None
            return False
        return True
    
    def _simple_convert(self, glosses: List[str]) -> str:
        """
        Rule-based conversion from glosses to English.
        Applies basic grammar rules to make output more readable.
        """
        if not glosses:
            return ""
        
        # Filter special tokens and normalize
        words = []
        for g in glosses:
            g_lower = g.lower()
            if g_lower not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                words.append(g_lower)
        
        if not words:
            return ""
        
        # Identify sentence type
        is_question = any(w in self.QUESTION_WORDS for w in words)
        
        # Extract time marker if at beginning
        time_phrase = ""
        if words and words[0] in self.TIME_MARKERS:
            time_phrase = words[0]
            words = words[1:]
        
        # Basic reconstruction
        sentence = " ".join(words)
        
        # Add articles where likely needed
        sentence = self._add_articles(sentence)
        
        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Add time phrase
        if time_phrase:
            sentence = time_phrase.capitalize() + ", " + sentence[0].lower() + sentence[1:]
        
        # Add punctuation
        if is_question:
            sentence = sentence.rstrip('.') + "?"
        elif sentence and not sentence.endswith(('.', '!', '?')):
            sentence += "."
        
        return sentence
    
    def _add_articles(self, text: str) -> str:
        """Add articles (a, an, the) where appropriate."""
        # Common nouns that typically need articles
        nouns_needing_article = {
            'doctor', 'meeting', 'train', 'bus', 'car', 'house', 'hospital',
            'school', 'office', 'shop', 'restaurant', 'hotel', 'airport',
            'station', 'bank', 'church', 'park', 'beach', 'movie', 'book',
            'phone', 'computer', 'problem', 'question', 'answer', 'message'
        }
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # Check if previous word is already an article or determiner
            if i > 0 and result[-1] in ['a', 'an', 'the', 'my', 'your', 'his', 'her', 'our', 'their']:
                result.append(word)
                continue
            
            # Add 'the' before certain nouns
            if word in nouns_needing_article:
                result.append('the')
            
            result.append(word)
        
        return " ".join(result)
    
    def _llm_convert(self, glosses: List[str]) -> str:
        """Use local FLAN-T5 for conversion."""
        gloss_str = " ".join(g.upper() for g in glosses 
                           if g.lower() not in ['<unk>', '<pad>', '<sos>', '<eos>'])
        
        prompt = f"""Convert these British Sign Language (BSL) glosses into a natural English sentence.
BSL glosses are written in capitals and follow BSL grammar (topic-comment, time first, no articles).

BSL Glosses: {gloss_str}

Natural English sentence:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                num_beams=3
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up result
        result = result.strip()
        if result and not result.endswith(('.', '!', '?')):
            result += "."
        
        return result if result else self._simple_convert(glosses)
    
    @staticmethod
    def _finalize_sentence(result: str) -> str:
        """Strip wrapping quotes and ensure terminal punctuation."""
        result = result.strip().strip('"\'')
        if result and not result.endswith(('.', '!', '?')):
            result += "."
        return result

    def _cerebras_convert(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call the Cerebras chat completions API (OpenAI-compatible).
        Returns the sentence, or None if no key is configured or the
        response has no content. Raises on HTTP errors and timeouts.
        """
        if not self.cerebras_api_key:
            return None

        import requests

        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.cerebras_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.cerebras_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_completion_tokens": 512,
                "reasoning_effort": "low",
            },
            timeout=self.cerebras_timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content")
        return content.strip() if content and content.strip() else None

    def _groq_convert(self, glosses: List[str]) -> str:
        """
        Hosted LLM conversion: Cerebras (gpt-oss-120b) primary,
        Groq (llama-3.3-70b-versatile) fallback.
        """
        gloss_str = " ".join(g.upper() for g in glosses
                           if g.lower() not in ['<unk>', '<pad>', '<sos>', '<eos>'])

        if not gloss_str.strip():
            return ""

        system_prompt = """You are an expert translator from British Sign Language (BSL) glosses to natural English.

BSL glosses are individual sign words written in CAPITALS. BSL grammar differs from English:
- Topic-comment structure (topic comes first, then comment)
- Time markers appear at the beginning
- No articles (a, an, the)
- No auxiliary verbs (is, are, was, were)
- Different word order

Your task: Convert the BSL glosses into a natural, grammatically correct English sentence.
- Add appropriate articles, pronouns, and auxiliary verbs
- Rearrange words to follow English grammar
- Keep the meaning intact
- Return ONLY the English sentence, nothing else."""

        user_prompt = f"BSL Glosses: {gloss_str}"

        # Primary: Cerebras
        try:
            result = self._cerebras_convert(system_prompt, user_prompt)
            if result:
                print("[LLM] served by: cerebras")
                return self._finalize_sentence(result)
            if self.cerebras_api_key:
                print("[LLM] Cerebras returned empty content, falling back to Groq")
        except Exception as e:
            print(f"[LLM] Cerebras error ({type(e).__name__}): {e}, falling back to Groq")

        # Fallback: Groq. Built on demand, so a run whose Cerebras calls all
        # succeed never needs the SDK present at all.
        if not self._init_groq_client():
            print("[LLM] No Groq fallback available, using rule-based output")
            return self._simple_convert(glosses)

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()
            print("[LLM] served by: groq (fallback)")
            return self._finalize_sentence(result)

        except Exception as e:
            # Both hosted providers are gone, so the sentence quality drops
            # sharply. Say so plainly rather than returning quietly.
            print(f"[LLM] Groq error ({type(e).__name__}): {e}, using rule-based output")
            return self._simple_convert(glosses)
    
    def convert(self, glosses: List[str]) -> str:
        """
        Convert BSL glosses to natural English text.
        
        Args:
            glosses: List of BSL glosses (strings)
            
        Returns:
            Natural English sentence
        """
        if not glosses:
            return ""
        
        # Filter empty strings
        glosses = [g for g in glosses if g and g.strip()]
        
        if not glosses:
            return ""
        
        if self.mode == 'simple':
            return self._simple_convert(glosses)
        elif self.mode == 'llm':
            return self._llm_convert(glosses)
        elif self.mode == 'groq':
            return self._groq_convert(glosses)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def convert_batch(self, gloss_sequences: List[List[str]]) -> List[str]:
        """
        Convert multiple gloss sequences.
        
        Args:
            gloss_sequences: List of gloss lists
            
        Returns:
            List of English sentences
        """
        return [self.convert(glosses) for glosses in gloss_sequences]


class BSLToSpeechPipeline:
    """
    Complete Direction 1 pipeline: BSL Glosses -> Text -> Speech
    
    Combines GlossToText with Coqui TTS for end-to-end conversion.
    """
    
    def __init__(
        self,
        gloss_mode: str = "simple",
        groq_api_key: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        tts_language: str = "en"
    ):
        """
        Initialize pipeline.
        
        Args:
            gloss_mode: Mode for gloss-to-text ('simple', 'llm', 'groq')
            groq_api_key: Groq API key (for mode='groq')
            speaker_wav: Reference voice for TTS
            tts_language: TTS language code
        """
        print("Initializing BSL-to-Speech pipeline...")
        
        # Initialize gloss-to-text converter
        self.gloss_to_text = GlossToText(
            mode=gloss_mode,
            groq_api_key=groq_api_key
        )
        
        # Initialize TTS if speaker_wav provided
        self.tts = None
        self.speaker_wav = speaker_wav
        self.tts_language = tts_language
        
        if speaker_wav and os.path.exists(speaker_wav):
            self._init_tts()
        
        print("BSL-to-Speech pipeline ready.")
    
    def _init_tts(self):
        """Initialize Coqui TTS."""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, TTS disabled")
            return
        
        try:
            from TTS.api import TTS
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading TTS model on {device}...")
            
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            print("TTS model loaded.")
            
        except ImportError:
            print("Warning: Coqui TTS not installed, TTS disabled")
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")
    
    def glosses_to_text(self, glosses: List[str]) -> str:
        """Convert glosses to natural English text."""
        return self.gloss_to_text.convert(glosses)
    
    def text_to_speech(self, text: str, output_path: str) -> Optional[str]:
        """Convert text to speech audio."""
        if not self.tts:
            print("TTS not initialized. Provide speaker_wav to enable.")
            return None
        
        if not text or not text.strip():
            print("Empty text, skipping TTS")
            return None
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.tts_language,
            file_path=output_path
        )
        
        return output_path
    
    def process(
        self,
        glosses: List[str],
        output_audio: Optional[str] = None
    ) -> Dict:
        """
        Process glosses through full pipeline.
        
        Args:
            glosses: List of BSL glosses
            output_audio: Path for output audio (optional)
            
        Returns:
            Dict with text and optional audio_path
        """
        # Convert glosses to text
        text = self.glosses_to_text(glosses)
        
        result = {
            "glosses": glosses,
            "text": text
        }
        
        # Generate speech if requested
        if output_audio and self.tts:
            audio_path = self.text_to_speech(text, output_audio)
            result["audio_path"] = audio_path
        
        return result


def demo():
    """Demonstrate gloss-to-text conversion."""
    print("=" * 60)
    print("Gloss-to-Text Converter Demo")
    print("=" * 60)
    
    # Test sequences (BSL gloss order)
    test_sequences = [
        ["TOMORROW", "MEETING", "WHAT", "TIME"],
        ["YESTERDAY", "I", "GO", "DOCTOR"],
        ["HELP", "LONDON", "LIVE", "FIND"],
        ["THANK", "YOU", "MUCH"],
        ["MY", "NAME", "SARAH"],
        ["WEATHER", "TODAY", "BEAUTIFUL"],
        ["TRAIN", "MANCHESTER", "LEAVE", "THREE", "AFTERNOON"],
        ["I", "NOT", "UNDERSTAND"],
    ]
    
    converter = GlossToText(mode="simple")
    
    print("\nSimple Mode (Rule-Based):")
    print("-" * 60)
    
    for glosses in test_sequences:
        gloss_str = " ".join(glosses)
        text = converter.convert(glosses)
        print(f"Glosses: {gloss_str}")
        print(f"English: {text}")
        print()


if __name__ == "__main__":
    demo()
