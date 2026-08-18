"""
Speech to BSL Pipeline (Direction 2)
Speech Audio → Text (Whisper) → BSL Glosses → Avatar

Author: BSL Translation Project
"""

import os
import json
import re
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Optional imports - only needed for full pipeline
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class WhisperASR:
    """
    Speech-to-Text using OpenAI Whisper.
    
    Converts audio input to English text transcription.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper ASR.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for WhisperASR. Install with: pip install torch")
            
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded.")
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "en",
        return_segments: bool = False
    ) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            language: Language code (default: 'en' for English)
            return_segments: If True, include word-level timestamps
            
        Returns:
            Dict with 'text' and optionally 'segments'
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=return_segments
        )
        
        output = {"text": result["text"].strip()}
        
        if return_segments:
            output["segments"] = result.get("segments", [])
        
        return output


class TextToGloss:
    """
    Convert English text to BSL gloss sequence.
    
    BSL has different grammar from English:
    - Topic-comment structure (topic comes first)
    - Time markers at the beginning
    - No articles (a, an, the)
    - Different word order
    
    Modes:
    - 'simple': Rule-based extraction (fast, basic)
    - 'llm': Local model (FLAN-T5)
    - 'groq': Groq API with Llama 3.1 (best quality)
    """
    
    # Common English words to remove (not typically signed)
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'it', 'its', "it's", 'that', 'which', 'who', 'whom', 'whose', 'this', 'these', 'those',
        'just', 'very', 'really', 'quite', 'rather', 'too', 'also', 'only'
    }
    
    # Number to word mapping for BSL glosses
    NUMBER_WORDS = {
        '0': 'ZERO', '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR',
        '5': 'FIVE', '6': 'SIX', '7': 'SEVEN', '8': 'EIGHT', '9': 'NINE',
        '10': 'TEN', '11': 'ELEVEN', '12': 'TWELVE', '13': 'THIRTEEN',
        '14': 'FOURTEEN', '15': 'FIFTEEN', '16': 'SIXTEEN', '17': 'SEVENTEEN',
        '18': 'EIGHTEEN', '19': 'NINETEEN', '20': 'TWENTY',
        '21': 'TWENTY-ONE', '22': 'TWENTY-TWO', '23': 'TWENTY-THREE',
        '24': 'TWENTY-FOUR', '25': 'TWENTY-FIVE', '26': 'TWENTY-SIX',
        '27': 'TWENTY-SEVEN', '28': 'TWENTY-EIGHT', '29': 'TWENTY-NINE',
        '30': 'THIRTY', '40': 'FORTY', '50': 'FIFTY', '60': 'SIXTY',
        '70': 'SEVENTY', '80': 'EIGHTY', '90': 'NINETY',
        '100': 'HUNDRED', '1000': 'THOUSAND', '1000000': 'MILLION',
    }
    
    # BSL time markers (placed at beginning of sentence)
    TIME_MARKERS = {
        'yesterday', 'today', 'tomorrow', 'now', 'later', 'before', 'after',
        'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        # Add number-related time words
        'oclock', "o'clock"
    }
    
    def __init__(
        self, 
        mode: str = "simple",
        vocabulary_path: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        strict_vocab: bool = False
    ):
        """
        Initialize Text-to-Gloss converter.
        
        Args:
            mode: Conversion mode ('simple', 'llm', 'groq')
            vocabulary_path: Path to vocabulary.json (for filtering valid glosses)
            groq_api_key: API key for Groq (required if mode='groq')
            strict_vocab: If True, only output glosses in vocabulary. 
                         If False (default), output all glosses but mark coverage.
        """
        self.mode = mode
        self.vocabulary = self._load_vocabulary(vocabulary_path)
        self.groq_api_key = groq_api_key
        self.strict_vocab = strict_vocab
        
        # Add number words to vocabulary if not present
        self._ensure_numbers_in_vocab()
        
        if self.vocabulary:
            print(f"Loaded vocabulary with {len(self.vocabulary)} glosses (strict={strict_vocab})")
        
        if mode == 'llm':
            self._init_local_llm()
        elif mode == 'groq':
            self._init_groq()
    
    def _load_vocabulary(self, vocab_path: Optional[str]) -> Optional[set]:
        """Load vocabulary for filtering valid glosses."""
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            # Handle different vocabulary formats
            if isinstance(vocab_data, dict):
                if 'gloss_to_idx' in vocab_data:
                    return set(vocab_data['gloss_to_idx'].keys())
                else:
                    return set(vocab_data.keys())
            elif isinstance(vocab_data, list):
                return set(vocab_data)
        return None
    
    def _ensure_numbers_in_vocab(self):
        """Ensure number glosses are in vocabulary."""
        if self.vocabulary is not None:
            number_glosses = set(self.NUMBER_WORDS.values())
            # Add compound numbers
            number_glosses.update(['TWENTY-ONE', 'TWENTY-TWO', 'TWENTY-THREE', 
                                   'TWENTY-FOUR', 'TWENTY-FIVE', 'TWENTY-SIX',
                                   'TWENTY-SEVEN', 'TWENTY-EIGHT', 'TWENTY-NINE'])
            # Add to vocabulary (lowercase to match existing format)
            for gloss in number_glosses:
                self.vocabulary.add(gloss.lower())
                self.vocabulary.add(gloss.upper())
    
    def _init_local_llm(self):
        """Initialize local FLAN-T5 model for text-to-gloss conversion."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LLM mode. Install with: pip install torch")
            
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )
        
        print("Loading FLAN-T5 model for text-to-gloss conversion...")
        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_model.to(device)
        self.llm_device = device
        print(f"FLAN-T5 loaded on {device}")
    
    def _init_groq(self):
        """Initialize Groq API client."""
        if not self.groq_api_key:
            raise ValueError("Groq API key required for mode='groq'")
        
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "Groq not installed. Install with: pip install groq"
            )
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        print("Groq API client initialized.")
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """
        Convert numeric digits to word equivalents before gloss conversion.
        
        Handles:
        - Single digits: 5 -> five
        - Two-digit numbers: 10 -> ten, 25 -> twenty-five
        - Time expressions: 5 o'clock -> five o'clock
        """
        # Sort by length (longest first) to handle larger numbers first
        for num, word in sorted(self.NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
            # Use word boundary to avoid partial matches
            text = re.sub(rf'\b{num}\b', word.lower(), text)
        
        # Handle remaining numbers (build compound numbers)
        def replace_number(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                if num < 100:
                    if num in [int(k) for k in self.NUMBER_WORDS.keys() if k.isdigit()]:
                        return self.NUMBER_WORDS.get(num_str, num_str).lower()
                    elif num < 20:
                        return self.NUMBER_WORDS.get(str(num), num_str).lower()
                    elif num < 100:
                        tens = (num // 10) * 10
                        ones = num % 10
                        tens_word = self.NUMBER_WORDS.get(str(tens), '')
                        if ones > 0:
                            ones_word = self.NUMBER_WORDS.get(str(ones), '')
                            return f"{tens_word.lower()}-{ones_word.lower()}"
                        return tens_word.lower()
                return num_str
            except ValueError:
                return num_str
        
        # Replace any remaining numbers
        text = re.sub(r'\b\d+\b', replace_number, text)
        
        return text
    
    def _lemmatize(self, word: str) -> str:
        """
        Simple rule-based lemmatization for common English verb/noun forms.
        Falls back to original word if no rule matches.
        """
        # Irregular verbs (most common)
        irregulars = {
            'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
            'had': 'have', 'has': 'have', 'having': 'have',
            'did': 'do', 'does': 'do', 'doing': 'do', 'done': 'do',
            'went': 'go', 'goes': 'go', 'going': 'go', 'gone': 'go',
            'came': 'come', 'comes': 'come', 'coming': 'come',
            'said': 'say', 'says': 'say', 'saying': 'say',
            'made': 'make', 'makes': 'make', 'making': 'make',
            'took': 'take', 'takes': 'take', 'taking': 'take', 'taken': 'take',
            'saw': 'see', 'sees': 'see', 'seeing': 'see', 'seen': 'see',
            'got': 'get', 'gets': 'get', 'getting': 'get', 'gotten': 'get',
            'gave': 'give', 'gives': 'give', 'giving': 'give', 'given': 'give',
            'found': 'find', 'finds': 'find', 'finding': 'find',
            'told': 'tell', 'tells': 'tell', 'telling': 'tell',
            'felt': 'feel', 'feels': 'feel', 'feeling': 'feel',
            'left': 'leave', 'leaves': 'leave', 'leaving': 'leave',
            'thought': 'think', 'thinks': 'think', 'thinking': 'think',
            'knew': 'know', 'knows': 'know', 'knowing': 'know', 'known': 'know',
            'wanted': 'want', 'wants': 'want', 'wanting': 'want',
            'used': 'use', 'uses': 'use', 'using': 'use',
            'tried': 'try', 'tries': 'try', 'trying': 'try',
            'called': 'call', 'calls': 'call', 'calling': 'call',
            'needed': 'need', 'needs': 'need', 'needing': 'need',
            'become': 'become', 'became': 'become', 'becomes': 'become',
            'began': 'begin', 'begins': 'begin', 'beginning': 'begin',
            'kept': 'keep', 'keeps': 'keep', 'keeping': 'keep',
            'let': 'let', 'lets': 'let', 'letting': 'let',
            'put': 'put', 'puts': 'put', 'putting': 'put',
            'ran': 'run', 'runs': 'run', 'running': 'run',
            'sat': 'sit', 'sits': 'sit', 'sitting': 'sit',
            'stood': 'stand', 'stands': 'stand', 'standing': 'stand',
            'lost': 'lose', 'loses': 'lose', 'losing': 'lose',
            'paid': 'pay', 'pays': 'pay', 'paying': 'pay',
            'met': 'meet', 'meets': 'meet', 'meeting': 'meet',
            'brought': 'bring', 'brings': 'bring', 'bringing': 'bring',
            'bought': 'buy', 'buys': 'buy', 'buying': 'buy',
            'led': 'lead', 'leads': 'lead', 'leading': 'lead',
            'held': 'hold', 'holds': 'hold', 'holding': 'hold',
            'wrote': 'write', 'writes': 'write', 'writing': 'write', 'written': 'write',
            'read': 'read', 'reads': 'read', 'reading': 'read',
            'spoke': 'speak', 'speaks': 'speak', 'speaking': 'speak', 'spoken': 'speak',
            'ate': 'eat', 'eats': 'eat', 'eating': 'eat', 'eaten': 'eat',
            'drank': 'drink', 'drinks': 'drink', 'drinking': 'drink', 'drunk': 'drink',
            'drove': 'drive', 'drives': 'drive', 'driving': 'drive', 'driven': 'drive',
            'lived': 'live', 'lives': 'live', 'living': 'live',
            'worked': 'work', 'works': 'work', 'working': 'work',
            'played': 'play', 'plays': 'play', 'playing': 'play',
            'moved': 'move', 'moves': 'move', 'moving': 'move',
            'liked': 'like', 'likes': 'like', 'liking': 'like',
            'loved': 'love', 'loves': 'love', 'loving': 'love',
            'started': 'start', 'starts': 'start', 'starting': 'start',
            'stopped': 'stop', 'stops': 'stop', 'stopping': 'stop',
            'helped': 'help', 'helps': 'help', 'helping': 'help',
            'asked': 'ask', 'asks': 'ask', 'asking': 'ask',
            'looked': 'look', 'looks': 'look', 'looking': 'look',
            'watched': 'watch', 'watches': 'watch', 'watching': 'watch',
            'talked': 'talk', 'talks': 'talk', 'talking': 'talk',
            'walked': 'walk', 'walks': 'walk', 'walking': 'walk',
            'waited': 'wait', 'waits': 'wait', 'waiting': 'wait',
            'turned': 'turn', 'turns': 'turn', 'turning': 'turn',
            'showed': 'show', 'shows': 'show', 'showing': 'show', 'shown': 'show',
            'heard': 'hear', 'hears': 'hear', 'hearing': 'hear',
            'learned': 'learn', 'learns': 'learn', 'learning': 'learn', 'learnt': 'learn',
            'changed': 'change', 'changes': 'change', 'changing': 'change',
            'followed': 'follow', 'follows': 'follow', 'following': 'follow',
            'created': 'create', 'creates': 'create', 'creating': 'create',
            'opened': 'open', 'opens': 'open', 'opening': 'open',
            'closed': 'close', 'closes': 'close', 'closing': 'close',
            # Plural nouns
            'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
            'feet': 'foot', 'teeth': 'tooth', 'mice': 'mouse',
        }
        
        if word in irregulars:
            return irregulars[word]
        
        # Regular patterns (order matters - check longer suffixes first)
        # -ies → -y (tries → try)
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        
        # -ied → -y (tried → try)
        if word.endswith('ied') and len(word) > 4:
            return word[:-3] + 'y'
        
        # -ving → -ve (living → live, moving → move)
        if word.endswith('ving') and len(word) > 5:
            return word[:-3] + 'e'
        
        # -ting with double consonant → single (stopping → stop, running → run)
        if word.endswith('ting') and len(word) > 5:
            if word[-5] == word[-6]:  # double consonant
                return word[:-4]
        
        # -ning with double consonant → single (running → run)
        if word.endswith('ning') and len(word) > 5:
            if word[-5] == word[-6]:
                return word[:-4]
        
        # -ing → base (walking → walk, talking → talk)
        if word.endswith('ing') and len(word) > 4:
            base = word[:-3]
            # Check if base + 'e' is more likely (making → make)
            if len(base) > 1 and base[-1] in 'kgc':
                return base + 'e'
            return base
        
        # -ed → base (walked → walk, talked → talk)
        if word.endswith('ed') and len(word) > 3:
            # -ied handled above
            # -xed, -sed, etc. → just remove -ed
            if word[-3] in 'aeiouxs':
                return word[:-2]
            # doubled consonant (stopped → stop)
            if len(word) > 4 and word[-3] == word[-4]:
                return word[:-3]
            # -ed after consonant (walked → walk)
            return word[:-2]
        
        # -es → base (watches → watch, goes → go)
        if word.endswith('es') and len(word) > 3:
            if word.endswith('ches') or word.endswith('shes') or word.endswith('sses') or word.endswith('xes'):
                return word[:-2]
            if word.endswith('ies'):
                return word[:-3] + 'y'
            return word[:-1]
        
        # -s → base (walks → walk) - but not words that naturally end in s
        if word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            return word[:-1]
        
        return word
    
    def _simple_convert(self, text: str) -> List[str]:
        """
        Rule-based conversion from English to BSL glosses.
        
        Process:
        1. Convert numbers to words
        2. Expand contractions
        3. Tokenize and lowercase
        4. Remove stop words
        5. Lemmatize words
        6. Extract time markers to front
        7. Filter to known vocabulary (if available and strict mode)
        """
        # Step 1: Convert numbers to words FIRST
        text = self._convert_numbers_to_words(text)
        
        # Expand common contractions
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "can not", "couldn't": "could not", "won't": "will not",
            "wouldn't": "would not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "i'm": "i am", "you're": "you are",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are", "i've": "i have",
            "you've": "you have", "we've": "we have", "they've": "they have",
            "i'll": "i will", "you'll": "you will", "he'll": "he will",
            "she'll": "she will", "we'll": "we will", "they'll": "they will",
            "i'd": "i would", "you'd": "you would", "he'd": "he would",
            "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "how's": "how is", "that's": "that is", "there's": "there is",
            "here's": "here is", "let's": "let us",
            "o'clock": "oclock"  # Handle o'clock specially
        }
        
        # Tokenize
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Handle hyphenated numbers (twenty-five -> TWENTY FIVE as separate glosses)
        text = text.replace('-', ' ')
        
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Separate time markers and lemmatize content words
        time_words = []
        content_words = []
        
        for word in words:
            # Check if it's a number word
            if word.upper() in self.NUMBER_WORDS.values():
                content_words.append(word.upper())
            elif word in self.TIME_MARKERS:
                time_words.append(word.upper())
            elif word not in self.STOP_WORDS:
                # Apply lemmatization
                lemma = self._lemmatize(word)
                content_words.append(lemma.upper())
        
        # BSL structure: TIME + TOPIC + COMMENT
        glosses = time_words + content_words
        
        # Filter to known vocabulary only if strict mode enabled
        if self.vocabulary and self.strict_vocab:
            glosses = [g for g in glosses if g.lower() in self.vocabulary or g in self.vocabulary]
        
        return glosses if glosses else ["<unk>"]
    
    def convert_with_info(self, text: str) -> Dict:
        """
        Convert text to glosses with vocabulary coverage info.
        
        Returns:
            Dict with:
                - glosses: List of all glosses
                - in_vocab: List of glosses that are in the vocabulary
                - out_of_vocab: List of glosses not in vocabulary
                - coverage: Percentage of glosses in vocabulary
        """
        glosses = self.convert(text)
        
        if not self.vocabulary:
            return {
                "glosses": glosses,
                "in_vocab": glosses,
                "out_of_vocab": [],
                "coverage": 100.0
            }
        
        in_vocab = []
        out_of_vocab = []
        
        for g in glosses:
            if g.lower() in self.vocabulary or g in self.vocabulary:
                in_vocab.append(g)
            else:
                out_of_vocab.append(g)
        
        coverage = (len(in_vocab) / len(glosses) * 100) if glosses else 0.0
        
        return {
            "glosses": glosses,
            "in_vocab": in_vocab,
            "out_of_vocab": out_of_vocab,
            "coverage": coverage
        }
    
    def _llm_convert(self, text: str) -> List[str]:
        """Use local FLAN-T5 to convert text to glosses."""
        # Convert numbers first
        text = self._convert_numbers_to_words(text)
        
        prompt = f"""Convert this English sentence to British Sign Language (BSL) glosses.
BSL glosses are individual sign words in capital letters, in BSL word order (topic-comment, time first).
Remove articles (a, an, the) and auxiliary verbs.

English: {text}
BSL Glosses:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(self.llm_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse output to gloss list
        glosses = [g.strip().upper() for g in result.replace(',', ' ').split() if g.strip()]
        
        # Filter to vocabulary only if strict mode enabled
        if self.vocabulary and self.strict_vocab:
            glosses = [g for g in glosses if g.lower() in self.vocabulary or g in self.vocabulary]
        
        return glosses if glosses else self._simple_convert(text)
    
    def _groq_convert(self, text: str) -> List[str]:
        """Use Groq API (Llama 3.1) to convert text to glosses."""
        # Convert numbers first
        text = self._convert_numbers_to_words(text)
        
        # Build vocabulary hint if available
        vocab_hint = ""
        if self.vocabulary:
            # Sample some vocabulary items as hints
            sample_vocab = list(self.vocabulary)[:50]
            vocab_hint = f"\n\nAvailable glosses include: {', '.join(sample_vocab[:20])}..."
        
        prompt = f"""Convert this English sentence to British Sign Language (BSL) gloss notation.

Rules for BSL glosses:
1. Use CAPITAL LETTERS for each gloss/sign
2. Remove articles (a, an, the) and most auxiliary verbs
3. Put TIME markers at the beginning
4. Use topic-comment structure (what you're talking about first, then the comment)
5. Numbers should be spelled out as words (e.g., FIVE, TEN, TWENTY)
6. Return ONLY the glosses separated by spaces, nothing else
{vocab_hint}

English: "{text}"

BSL Glosses:"""

        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert in British Sign Language linguistics. Convert English to BSL gloss notation accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse output - allow hyphens for compound numbers
        glosses = [g.strip().upper() for g in result.replace(',', ' ').split() if g.strip()]
        
        # Filter to vocabulary only if strict mode enabled
        if self.vocabulary and self.strict_vocab:
            filtered = [g for g in glosses if g.lower() in self.vocabulary or g in self.vocabulary]
            if filtered:
                glosses = filtered
        
        return glosses if glosses else self._simple_convert(text)
    
    def convert(self, text: str) -> List[str]:
        """
        Convert English text to BSL glosses.
        
        Args:
            text: English sentence
            
        Returns:
            List of BSL glosses (uppercase strings)
        """
        if not text or not text.strip():
            return []
        
        if self.mode == 'simple':
            return self._simple_convert(text)
        elif self.mode == 'llm':
            return self._llm_convert(text)
        elif self.mode == 'groq':
            return self._groq_convert(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class GlossRenderer:
    """
    Render BSL glosses for display.
    
    Future: This will interface with a 3D avatar system.
    Current: Returns structured data for UI rendering.
    """
    
    def __init__(
        self, 
        gloss_video_dir: Optional[str] = None,
        gloss_duration: float = 1.0
    ):
        """
        Initialize gloss renderer.
        
        Args:
            gloss_video_dir: Directory containing gloss video clips (optional)
            gloss_duration: Default duration per gloss in seconds
        """
        self.gloss_video_dir = gloss_video_dir
        self.gloss_duration = gloss_duration
        self.gloss_videos = self._scan_videos()
    
    def _scan_videos(self) -> Dict[str, str]:
        """Scan for available gloss video clips."""
        if not self.gloss_video_dir or not os.path.exists(self.gloss_video_dir):
            return {}
        
        videos = {}
        for ext in ['*.mp4', '*.webm', '*.gif']:
            for path in Path(self.gloss_video_dir).glob(ext):
                gloss = path.stem.upper()
                videos[gloss] = str(path)
        
        return videos
    
    def render(self, glosses: List[str]) -> Dict:
        """
        Prepare gloss sequence for rendering.
        
        Args:
            glosses: List of BSL glosses
            
        Returns:
            Dict with rendering instructions for each gloss
        """
        timeline = []
        current_time = 0.0
        
        for gloss in glosses:
            entry = {
                "gloss": gloss,
                "start_time": current_time,
                "end_time": current_time + self.gloss_duration,
                "duration": self.gloss_duration,
            }
            
            # Check for video clip
            if gloss in self.gloss_videos:
                entry["video_path"] = self.gloss_videos[gloss]
                entry["render_type"] = "video"
            else:
                entry["render_type"] = "text"  # Fallback to text display
            
            timeline.append(entry)
            current_time += self.gloss_duration
        
        return {
            "glosses": glosses,
            "total_duration": current_time,
            "timeline": timeline,
            "available_videos": len([t for t in timeline if t["render_type"] == "video"]),
            "missing_videos": len([t for t in timeline if t["render_type"] == "text"])
        }


class CoquiTTS:
    """
    Text-to-Speech using Coqui TTS with XTTS v2.
    Supports voice cloning from reference audio.
    """
    
    def __init__(
        self,
        speaker_wav: str,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
        device: Optional[str] = None
    ):
        """
        Initialize Coqui TTS.
        
        Args:
            speaker_wav: Path to reference voice audio file
            model_name: TTS model identifier
            language: Language code
            device: Device for inference (cuda/cpu)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Coqui TTS not installed. Install with: pip install TTS")
        
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav}")
        
        self.speaker_wav = speaker_wav
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading TTS model on {self.device}...")
        self.tts = TTS(model_name).to(self.device)
        print("TTS model loaded.")
    
    def synthesize(self, text: str, output_path: str) -> str:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to synthesize
            output_path: Path for output audio file
            
        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.language,
            file_path=output_path
        )
        
        return output_path
    
    def synthesize_glosses(self, glosses: List[str], output_path: str) -> str:
        """
        Convert BSL glosses to speech.
        Joins glosses into a sentence before synthesis.
        
        Args:
            glosses: List of BSL glosses
            output_path: Path for output audio file
            
        Returns:
            Path to generated audio file
        """
        # Convert glosses to readable sentence
        text = " ".join(g.lower() for g in glosses if g not in ['<unk>', '<pad>', '<sos>', '<eos>'])
        return self.synthesize(text, output_path)


class SpeechToBSL:
    """
    Complete Speech-to-BSL pipeline (Direction 2).
    
    Speech → Whisper → English Text → Text-to-Gloss → BSL Glosses → Renderer
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        gloss_mode: str = "simple",
        vocabulary_path: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gloss_video_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Speech-to-BSL pipeline.
        
        Args:
            whisper_model: Whisper model size
            gloss_mode: Text-to-gloss conversion mode ('simple', 'llm', 'groq')
            vocabulary_path: Path to vocabulary.json
            groq_api_key: Groq API key (if using mode='groq')
            gloss_video_dir: Directory with gloss video clips
            device: Device for inference
        """
        print("Initializing Speech-to-BSL pipeline...")
        
        # Initialize components
        self.asr = WhisperASR(model_size=whisper_model, device=device)
        self.text_to_gloss = TextToGloss(
            mode=gloss_mode,
            vocabulary_path=vocabulary_path,
            groq_api_key=groq_api_key
        )
        self.renderer = GlossRenderer(gloss_video_dir=gloss_video_dir)
        
        print("Speech-to-BSL pipeline ready.")
    
    def process(
        self, 
        audio_path: str,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Process speech audio to BSL output.
        
        Args:
            audio_path: Path to audio file
            return_intermediate: Include intermediate results (text, etc.)
            
        Returns:
            Dict with BSL glosses and rendering data
        """
        # Step 1: Speech to Text
        asr_result = self.asr.transcribe(audio_path)
        text = asr_result["text"]
        
        # Step 2: Text to Glosses
        glosses = self.text_to_gloss.convert(text)
        
        # Step 3: Prepare for rendering
        render_data = self.renderer.render(glosses)
        
        result = {
            "glosses": glosses,
            "render": render_data
        }
        
        if return_intermediate:
            result["text"] = text
            result["asr_result"] = asr_result
        
        return result
    
    def process_text(self, text: str) -> Dict:
        """
        Process text directly (skip ASR).
        
        Useful for testing or when text is already available.
        """
        glosses = self.text_to_gloss.convert(text)
        render_data = self.renderer.render(glosses)
        
        return {
            "text": text,
            "glosses": glosses,
            "render": render_data
        }


# ============================================================================
# Demo / Testing
# ============================================================================

def demo_simple():
    """Demo text-to-gloss with number handling."""
    print("=" * 60)
    print("Speech-to-BSL Pipeline Demo (with Number Support)")
    print("=" * 60)
    
    # Test sentences including numbers
    test_sentences = [
        "Hello, my name is John.",
        "What time is the meeting tomorrow?",
        "5 o'clock, 6 o'clock, 7 o'clock, 10 o'clock",
        "I have 3 children and 2 dogs.",
        "The meeting is at 10:30.",
        "She is 25 years old.",
        "I need 100 dollars.",
        "Can you help me please?",
        "Yesterday I went to the doctor.",
    ]
    
    # Initialize text-to-gloss converter
    converter = TextToGloss(mode="simple")
    
    print("\nText-to-Gloss Conversion (with Numbers):")
    print("-" * 60)
    
    for sentence in test_sentences:
        glosses = converter.convert(sentence)
        gloss_str = " ".join(glosses)
        print(f"Input:  {sentence}")
        print(f"Output: {gloss_str}")
        print()
    
    print("=" * 60)
    print("Number conversion working!")
    print("=" * 60)


if __name__ == "__main__":
    demo_simple()