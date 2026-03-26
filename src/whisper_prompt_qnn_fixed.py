#ENCODER_MODEL = "encoder_model.onnx"
#DECODER_MODEL = "decoder_model.onnx"
ENCODER_MODEL = "model.onnx"
DECODER_MODEL = "model.onnx"
#MODEL_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper/"
ENCODER_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper/whispersmallencoderquant"
DECODER_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper/whispersmalldecoderquant"
RECORD_FILE = "recording.wav"
# This script demonstrates how to use a QNN-optimized Whisper model with ONNX Runtime
# for speech-to-text transcription, including live audio with VAD. It supports
# three decoder modes:
#  - PLAIN (encoder_hidden_states, no caches)
#  - MERGED (past_key_values + use_cache_branch)
#  - QNN_KV (explicit uint8 self/cross KV caches + uint16 attention_mask + int32 position_ids)

#!/usr/bin/env python3

import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print("Imported from:", ort.__file__)
print(ort.get_available_providers())

import numpy as np
import sounddevice as sd
import webrtcvad
import os
import wave
import time
import whisper

# ===== Model locations (update as needed) =====
# Using separate quantized encoder/decoder folders as in your latest script.
ENCODER_MODEL = "model.onnx"
DECODER_MODEL = "model.onnx"
ENCODER_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper/whispersmallencoderquant"
DECODER_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper/whispersmalldecoderquant"

# ===== Provider configuration =====
# QNN configuration (currently disabled due to data type issues)
qnn_opts = {
    "backend_type": "htp",
    "fallback_on_unsupported_ops": "1",
    "fallback_on_error": "1"
}
print("QNN provider options:", qnn_opts)

# Using CPU provider only to avoid data type mismatch errors
# If you want to try QNN again, comment the line below and uncomment the next line
providers = ["CPUExecutionProvider"]
# providers = [("QNNExecutionProvider", qnn_opts), ("CPUExecutionProvider", {})]

# ===== VAD / streaming params =====
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 20
CHUNK = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT_MS = 800
SILENCE_TIMEOUT_FRAMES = max(1, SILENCE_TIMEOUT_MS // FRAME_DURATION_MS)
MAX_RECORD_SECONDS = 12
_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


class WhisperONNXRunner:
    """
    Supports three decoder styles:
      - PLAIN:   decoder expects input_ids + encoder_hidden_states
      - MERGED:  decoder expects input_ids + encoder_hidden_states + past_key_values.* + use_cache_branch
      - QNN_KV:  decoder expects input_ids(int32[1,1]), position_ids(int32[1]),
                 attention_mask(uint16[1,1,1,T_MAX]), self KV (uint8) per layer, cross KV (uint8) per layer

    Uses Whisper's preprocessing (OpenAI reference) for log-mel features and Whisper tokenizer for post.
    """

    def __init__(self):
        # Locate ONNX files
        enc_path = os.path.join(ENCODER_DIR_PATH, ENCODER_MODEL)
        dec_path = os.path.join(DECODER_DIR_PATH, DECODER_MODEL)
        if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            raise FileNotFoundError(
                f"Missing ONNX files. Encoder: {enc_path}  Decoder: {dec_path}")

        print("🧩 Loading encoder + decoder")
        options = ort.SessionOptions()
        self.encoder_sess = ort.InferenceSession(enc_path, sess_options=options, providers=providers)
        self.decoder_sess = ort.InferenceSession(dec_path, sess_options=options, providers=providers)
        actual_providers = self.encoder_sess.get_providers()
        print(f"✅ ONNX Runtime providers: {actual_providers}")
        print("ONNX models and processor loaded successfully.")

        # Tokenizer (English-only)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        self.eot_id = self.tokenizer.eot
        self.sot_seq = list(self.tokenizer.sot_sequence_including_notimestamps)

        # Encoder I/O
        self.enc_in = self.encoder_sess.get_inputs()[0].name  # usually 'input_features'

        # Debug print of model contracts
        print("--- Decoder inputs ---")
        for i in self.decoder_sess.get_inputs():
            print(i.name, i.shape, i.type)
        print("--- Encoder outputs ---")
        for i in self.encoder_sess.get_outputs():
            print(i.name, i.shape, i.type)

        # --- Decoder I/O & mode detection ---
        self.dec_inputs_list = list(self.decoder_sess.get_inputs())
        self.dec_inputs = {i.name: i for i in self.dec_inputs_list}
        self.dec_in_ids = next((i.name for i in self.dec_inputs_list if "input_ids" in i.name), None)

        # Legacy styles
        self.dec_in_enc = next((i.name for i in self.dec_inputs_list if "encoder_hidden_states" in i.name), None)
        self.dec_use_branch = next((i.name for i in self.dec_inputs_list if "use_cache_branch" in i.name), None)
        self.past_in_names = [i.name for i in self.dec_inputs_list if i.name.startswith("past_key_values")]

        # QNN KV style
        self.self_kv_in_names = sorted(
            [i.name for i in self.dec_inputs_list if i.name.startswith("k_cache_self_") and i.name.endswith("_in")] +
            [i.name for i in self.dec_inputs_list if i.name.startswith("v_cache_self_") and i.name.endswith("_in")] )
        self.cross_kv_names = sorted(
            [i.name for i in self.dec_inputs_list if i.name.startswith("k_cache_cross_")] +
            [i.name for i in self.dec_inputs_list if i.name.startswith("v_cache_cross_")] )
        self.attn_mask_name = next((i.name for i in self.dec_inputs_list if "attention_mask" in i.name), None)
        self.pos_ids_name   = next((i.name for i in self.dec_inputs_list if "position_ids" in i.name), None)

        if self.dec_in_ids is None:
            raise RuntimeError(f"Decoder missing input_ids; found {list(self.dec_inputs.keys())}")

        if self.dec_in_enc is not None:
            self.dec_mode = "PLAIN_OR_MERGED"
        else:
            # No encoder_hidden_states: expect explicit KV inputs
            if not self.self_kv_in_names or not self.cross_kv_names or not self.attn_mask_name or not self.pos_ids_name:
                raise RuntimeError(f"Unsupported decoder signature; found {list(self.dec_inputs.keys())}")
            self.dec_mode = "QNN_KV"

        self.dec_outs_list = list(self.decoder_sess.get_outputs())
        self.dec_out_names = [o.name for o in self.dec_outs_list]
        self.dec_logits = self.dec_out_names[0]
        print("🔧 Decoder mode:", self.dec_mode)
        print("Decoder outputs:", self.dec_out_names)

    # ===== Helpers =====
    def _np_dtype(self, onnx_type: str):
        t = (onnx_type or "").lower()
        if "int32" in t:   return np.int32
        if "int64" in t:   return np.int64
        if "uint8" in t:   return np.uint8
        if "uint16" in t:  return np.uint16
        if "float16" in t: return np.float16
        if "float" in t:   return np.float32
        if "bool" in t:    return np.bool_
        return np.float32

    def _zeros_like_input_shape(self, inp):
        # Produce zeros with exact static shape & dtype as declared by ONNX input
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        return np.zeros(shape, dtype=self._np_dtype(inp.type))

    def _mel(self, audio_path: str) -> np.ndarray:
        # OpenAI Whisper reference preprocessing
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel_t = whisper.log_mel_spectrogram(audio)  # torch.Tensor [80, 3000]
        mel_np = mel_t.detach().float().cpu().numpy()
        return np.expand_dims(mel_np, 0)  # [1, 80, 3000]

    def _update_past_from_outputs(self, out_names, out_list):
        # Map decoder self KV outputs back to the required *_in names
        next_kv = {}
        for name, arr in zip(out_names, out_list):
            if name.startswith("k_cache_self_") or name.startswith("v_cache_self_"):
                in_name = name.replace("_out", "_in") if name.endswith("_out") else name
                if in_name in self.dec_inputs:
                    next_kv[in_name] = np.ascontiguousarray(arr)
        return next_kv

    def _debug_print_feed(self, feed):
        try:
            print("\n=== FEED DICTIONARY DEBUG ===")
            for k, v in feed.items():
                if hasattr(v, "dtype"):
                    print(f"  {k}: dtype={v.dtype}, shape={getattr(v, 'shape', None)}")
                    # Print expected type if this is a model input
                    if k in self.dec_inputs:
                        expected_type = self._np_dtype(self.dec_inputs[k].type)
                        if v.dtype != expected_type:
                            print(f"    ⚠️ MISMATCH! Expected: {expected_type}")
                else:
                    print(f"  {k}: (non-array) {type(v)}")
            print("============================\n")
        except Exception as e:
            print(f"Error in debug_print_feed: {e}")
            
    def _ensure_correct_types(self, feed):
        """Ensure all inputs have the correct data types based on model expectations"""
        corrected_feed = {}
        
        for name, value in feed.items():
            if name not in self.dec_inputs:
                corrected_feed[name] = value
                continue
                
            expected_type = self._np_dtype(self.dec_inputs[name].type)
            
            if hasattr(value, "dtype") and value.dtype != expected_type:
                print(f"Converting {name} from {value.dtype} to {expected_type}")
                corrected_feed[name] = np.ascontiguousarray(value.astype(expected_type))
            else:
                corrected_feed[name] = value
                
        return corrected_feed

    # ===== Main API =====
    def transcribe(self, audio_path: str, max_tokens: int = 448, skip_special_tokens: bool = True) -> str:
        # 1) Pre: mel features
        mel = self._mel(audio_path)

        if self.dec_mode == "PLAIN_OR_MERGED":
            # Encoder → hidden states (single tensor)
            enc_hidden = self.encoder_sess.run(None, {self.enc_in: mel})[0]

            # Decoder loop (greedy)
            tokens = self.sot_seq[:]
            # Detect merged vs plain based on presence of branch/past inputs
            is_plain = (self.dec_use_branch is None and len(self.past_in_names) == 0)

            if is_plain:
                while len(tokens) < max_tokens:
                    input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
                    feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
                    dec_out = self.decoder_sess.run(self.dec_out_names, feed)
                    logits = dec_out[0]
                    next_id = int(np.argmax(logits[0, -1]))
                    tokens.append(next_id)
                    if next_id == self.eot_id:
                        break
            else:
                # First step with empty past
                input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
                feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
                # Initialize merged past_* and branch/cache position if present
                for inp in self.dec_inputs_list:
                    if inp.name.startswith("past_key_values"):
                        feed[inp.name] = self._zeros_like_input_shape(inp)
                if self.dec_use_branch is not None:
                    branch_dtype = self._np_dtype(self.dec_inputs[self.dec_use_branch].type)
                    feed[self.dec_use_branch] = np.asarray([0], dtype=branch_dtype)
                if self.dec_cache_pos is not None:
                    pos_dtype = self._np_dtype(self.dec_inputs[self.dec_cache_pos].type)
                    feed[self.dec_cache_pos] = np.asarray([0], dtype=pos_dtype)

                dec_out = self.decoder_sess.run(self.dec_out_names, feed)
                logits = dec_out[0]
                past = {}
                # Map present.* to past_key_values.* if that convention exists
                for name, arr in zip(self.dec_out_names, dec_out):
                    if name.startswith("present."):
                        past_name = name.replace("present.", "past_key_values.")
                        past[past_name] = arr

                # Subsequent steps reusing past
                branch_dtype = self._np_dtype(self.dec_inputs[self.dec_use_branch].type) if self.dec_use_branch else np.int64
                use_past_arr = np.asarray([1], dtype=branch_dtype)

                while len(tokens) < max_tokens:
                    next_id = int(np.argmax(logits[0, -1]))
                    tokens.append(next_id)
                    if next_id == self.eot_id:
                        break
                    input_ids = np.asarray([next_id], dtype=np.int64).reshape(1, -1)
                    feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
                    if self.dec_use_branch is not None:
                        feed[self.dec_use_branch] = use_past_arr
                    feed.update(past)
                    dec_out = self.decoder_sess.run(self.dec_out_names, feed)
                    logits = dec_out[0]
                    past = {k: v for k, v in past.items()}  # keep structure

            return self.tokenizer.decode(tokens).strip()

        # ===== QNN_KV path =====
        # 2) Encoder → returns cross-KV arrays (k_cache_cross_*, v_cache_cross_*)
        enc_out_list  = self.encoder_sess.run(None, {self.enc_in: mel})
        enc_out_names = [o.name for o in self.encoder_sess.get_outputs()]
        
        # Process encoder outputs and ensure correct types for cross-KV arrays
        enc_dict = {}
        for name, arr in zip(enc_out_names, enc_out_list):
            # Check if this output will be used as decoder input
            if name in self.cross_kv_names:
                # For cross KV caches, ensure they have the right type
                if name in self.dec_inputs:
                    expected_type = self._np_dtype(self.dec_inputs[name].type)
                    if arr.dtype != expected_type:
                        print(f"Converting encoder output {name} from {arr.dtype} to {expected_type}")
                        arr = arr.astype(expected_type)
            enc_dict[name] = np.ascontiguousarray(arr)

        # 3) Decoder loop (greedy) with explicit KV caches
        tokens = self.sot_seq[:]             # feed SOT one token per step

        # --- FORCE the expected QNN dtypes regardless of introspection ---
        ATTN_DTYPE = np.uint16   # attention_mask must be uint16
        POS_DTYPE  = np.int32    # position_ids must be int32
        IDS_DTYPE  = np.int32    # input_ids must be int32

        # T_MAX from model signature; fall back to 200 if symbolic
        T_MAX = self.dec_inputs[self.attn_mask_name].shape[-1]
        if not isinstance(T_MAX, int) or T_MAX <= 0:
            T_MAX = 200

        def _make_attn_mask(t_plus_1: int):
            m = np.zeros((1,1,1,T_MAX), dtype=ATTN_DTYPE)
            m[..., :t_plus_1] = 1
            # Explicitly ensure uint16 type
            if m.dtype != np.uint16:
                print(f"Warning: attention_mask dtype was {m.dtype}, converting to uint16")
                m = m.astype(np.uint16)
            return np.ascontiguousarray(m)

        def _zero_self_kv_feed():
            feed = {}
            for name in self.self_kv_in_names:
                inp = self.dec_inputs[name]
                z = self._zeros_like_input_shape(inp)
                # Ensure uint8 for QNN KV caches
                if z.dtype != np.uint8:
                    z = z.astype(np.uint8, copy=False)
                feed[name] = np.ascontiguousarray(z)
            return feed

        # ---- First step (t = 0) ----
        t = 0
        cur_token = np.array([[tokens[0]]], dtype=IDS_DTYPE)   # [1,1] int32
        position_ids = np.array([t], dtype=POS_DTYPE)          # [1]   int32
        attention_mask = _make_attn_mask(t + 1)                # [1,1,1,T_MAX] uint16

        # Force attention_mask to be uint16
        attention_mask = attention_mask.astype(np.uint16)
        
        feed = {
            self.dec_in_ids: np.ascontiguousarray(cur_token),
            self.attn_mask_name: attention_mask,
            self.pos_ids_name: np.ascontiguousarray(position_ids),
        }
        # zero self-KV
        feed.update(_zero_self_kv_feed())
        # add cross-KV from encoder (static for utterance)
        for name in self.cross_kv_names:
            feed[name] = np.ascontiguousarray(enc_dict[name])

        # Debug and ensure correct types
        self._debug_print_feed(feed)
        feed = self._ensure_correct_types(feed)
        dec_out = self.decoder_sess.run(self.dec_out_names, feed)
        logits = dec_out[0]
        next_self_kv = self._update_past_from_outputs(self.dec_out_names, dec_out)

        # ---- Continue through SOT prompt, then generate ----
        sot_tail = tokens[1:]
        generated = tokens[:]

        while len(generated) < max_tokens:
            next_id = int(np.argmax(logits[0, -1]))
            generated.append(next_id)
            if next_id == self.eot_id:
                break

            t += 1
            step_token = sot_tail.pop(0) if sot_tail else next_id

            cur_token = np.array([[step_token]], dtype=IDS_DTYPE)  # [1,1] int32
            position_ids = np.array([t], dtype=POS_DTYPE)          # [1]   int32
            attention_mask = _make_attn_mask(t + 1)

            # Force attention_mask to be uint16
            attention_mask = attention_mask.astype(np.uint16)
            
            feed = {
                self.dec_in_ids: np.ascontiguousarray(cur_token),
                self.attn_mask_name: attention_mask,
                self.pos_ids_name: np.ascontiguousarray(position_ids),
                **next_self_kv,
            }
            for name in self.cross_kv_names:
                feed[name] = np.ascontiguousarray(enc_dict[name])

            # Debug and ensure correct types
            self._debug_print_feed(feed)
            feed = self._ensure_correct_types(feed)
            dec_out = self.decoder_sess.run(self.dec_out_names, feed)
            logits = dec_out[0]
            next_self_kv = self._update_past_from_outputs(self.dec_out_names, dec_out)

        return self.tokenizer.decode(generated).strip()


# ===== Recording helpers (unchanged structure) =====

def _list_devices():
    devices = sd.query_devices()
    lines = []
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            lines.append(f"[{idx}] {d['name']} - inputs: {d['max_input_channels']}, samplerate: {int(d['default_samplerate'])}")
    return "".join(lines) if lines else "No input devices found."


def _record_audio_chunk_sd(device=None):
    try:
        sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    except Exception as e:
        print("Device check failed:", e)
        print("Available input devices:", _list_devices())
        return None

    print(" Listening for speech to start…")
    frames_bytes = []
    speaking = False
    silence_frames = 0
    total_frames = 0
    max_frames = int(MAX_RECORD_SECONDS * 1000 / FRAME_DURATION_MS)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK, channels=CHANNELS, dtype='int16', device=device) as stream:
        while True:
            data, overflowed = stream.read(CHUNK)  # raw bytes (int16 mono)
            is_speech = _vad.is_speech(data, SAMPLE_RATE)
            if not speaking:
                if is_speech:
                    print(" Speech detected. Recording…")
                    speaking = True
                    frames_bytes.append(data)
                    continue
            frames_bytes.append(data)
            silence_frames = 0 if is_speech else (silence_frames + 1)
            total_frames += 1
            if silence_frames > SILENCE_TIMEOUT_FRAMES:
                print(f" Detected {(silence_frames * FRAME_DURATION_MS) / 1000:.2f}s of silence. Chunk finished.")
                break
            if total_frames >= max_frames:
                print(" Maximum recording time reached. Stopping chunk.")
                break

    if not frames_bytes:
        print(" No speech was recorded in this chunk.")
        return None

    temp_file = f"recorded_chunk_{int(time.time())}.wav"
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames_bytes))
    print(f" Audio chunk saved to '{temp_file}'")
    return temp_file


def _record_and_transcribe_chunk(runner, device=None):
    audio_path = _record_audio_chunk_sd(device=device)
    if not audio_path:
        return None
    try:
        text = runner.transcribe(audio_path)
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
    return text


def wait_for_prompt(trigger_word: str, runner, device=None):
    print(f"--- Awaiting Trigger Word: '{trigger_word.upper()}' ---")
    try:
        while True:
            prompt = _record_and_transcribe_chunk(runner, device=device)
            if prompt:
                if trigger_word.lower() in prompt.lower():
                    print("!!! Trigger Word Detected !!!")
                    print(f"Full Prompt: {prompt.strip()}")
                    return prompt.strip()
                else:
                    print(f"Prompt thrown out (no '{trigger_word}' detected). Re-listening.")
                    print("-" * 30)
            else:
                print("No clear speech detected. Re-listening.")
                print("-" * 30)
    except KeyboardInterrupt:
        print("Listening stopped by user (KeyboardInterrupt).")
        return None


def get_question(runner, device=None):
    try:
        while True:
            prompt = _record_and_transcribe_chunk(runner, device=device)
            if prompt is None:
                print("No prompt captured. Re-listening…")
                continue
            print(f"Full Prompt: {prompt.strip()}")
            return prompt
    except KeyboardInterrupt:
        print("Listening stopped by user (KeyboardInterrupt).")
        return None


def main():
    runner = WhisperONNXRunner()
    prompt = get_question(runner, device=None)
    if prompt:
        print(f"Final Prompt Received: {prompt}")
    else:
        print("No prompt received.")


if __name__ == '__main__':
    main()
